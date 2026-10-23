import {useEffect,useMemo,useRef,useState} from 'react';
import Icon from './UiIcon';
import {bytes,mapGeometry,memoryBlockColor,memoryBlockTextColor,memorySpaceDescription,memorySpaceLabel} from './inspector-data';

const AXIS_WIDTH=110;
const HEADER_HEIGHT=48;
const SOURCE_LINE_HEIGHT=17;

function mergeIntervals(boxes){
 const intervals=boxes.map(box=>[box.start,box.end]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 return intervals.reduce((merged,[start,end])=>{
  const previous=merged.at(-1);
  if(previous&&start<=previous.end+1)previous.end=Math.max(previous.end,end);
  else merged.push({start,end});
  return merged;
 },[]);
}

// Each function keeps its source-line identity while inactive gaps are packed.
// The default 17 px scale matches the source inspector instead of collapsing
// one-line allocations into swimlane-like rules.
export function verticalProjection(fn,entries,lineScale=SOURCE_LINE_HEIGHT){
 const grouped=new Map();
 entries.forEach(entry=>{
  const name=entry.box.functionName||fn.name;
  if(!grouped.has(name))grouped.set(name,[]);
  grouped.get(name).push(entry.box);
 });
 const groups=[...grouped].map(([name,boxes])=>({name,boxes,segments:mergeIntervals(boxes)}));
 const scale=Math.max(.25,Math.min(51,lineScale)),factor=scale/SOURCE_LINE_HEIGHT;
 const positions=new Map(),labels=[];
 let y=8*factor;
 groups.forEach((group,groupIndex)=>{
  labels.push({name:group.name,y:y+10*factor});
  y+=18*factor;
  group.segments.forEach((segment,segmentIndex)=>{
   segment.y=y;
   const length=segment.end-segment.start+1;
   const tickStep=Math.max(5,Math.ceil(length/4/5)*5);
   labels.push({line:segment.start,y:y+10*factor});
   for(let line=Math.ceil(segment.start/tickStep)*tickStep;line<=segment.end;line+=tickStep){
    if(line!==segment.start)labels.push({line,y:y+(line-segment.start)*scale+10*factor});
   }
   group.boxes.forEach(box=>{
    if(box.start>=segment.start&&box.end<=segment.end&&!positions.has(box))positions.set(box,{y:y+(box.start-segment.start)*scale,height:Math.max(1,(box.end-box.start+1)*scale-factor)});
   });
   y+=length*scale;
   if(segmentIndex<group.segments.length-1)y+=9*factor;
  });
  if(groupIndex<groups.length-1)y+=14*factor;
 });
 return {height:Math.max(1,y+8*factor),labels,positions,lineScale:scale};
}

function truncate(name){return name.length>15?`${name.slice(0,14)}…`:name;}
function clamp(value,min,max){return Math.max(min,Math.min(max,value));}
function clampOffset(value,viewportSize,contentSize){return contentSize<=viewportSize?(viewportSize-contentSize)/2:Math.min(0,Math.max(viewportSize-contentSize,value));}

export function zoomViewport(view,factor,point,minScaleY){
 const scaleX=clamp(view.scaleX*factor,1,3),scaleY=clamp(view.scaleY*factor,minScaleY,3);
 return {scaleX,scaleY,x:point.x-(point.x-view.x)*scaleX/view.scaleX,y:point.y-(point.y-view.y)*scaleY/view.scaleY};
}

export default function FunctionMemoryMap({fn,spaceName,fitUsed,colorMode,selected,line,onSelect}){
 const host=useRef(null),plot=useRef(null),pan=useRef(null),limitsRef=useRef({minScaleY:1}),viewRef=useRef({x:0,y:0,scaleX:1,scaleY:1});
 const [viewport,setViewport]=useState({width:1000,height:700}),[view,setView]=useState(viewRef.current);
 const moveView=next=>{viewRef.current=next;setView(next);};
 useEffect(()=>{const observer=new ResizeObserver(entries=>{const {width,height}=entries[0].contentRect;setViewport({width,height});});observer.observe(host.current);return()=>observer.disconnect();},[]);

 const entries=useMemo(()=>fn.boxes.map((box,index)=>({box,index})).filter(({box})=>spaceName==='all'||box.space===spaceName),[fn,spaceName]);
 const spaces=fn.spaces.filter(item=>entries.some(({box})=>box.space===item.space));
 const plotWidth=Math.max(viewport.width-AXIS_WIDTH,120),plotHeight=Math.max(viewport.height-HEADER_HEIGHT,120);
 const baseProjection=verticalProjection(fn,entries,SOURCE_LINE_HEIGHT);
 const minScaleY=Math.min(1,plotHeight/baseProjection.height);
 limitsRef.current.minScaleY=minScaleY;
 const contentWidth=plotWidth*view.scaleX;
 const spans=spaces.map(space=>({space,span:(fitUsed?space.hwm:space.limit)||1}));
 const weightTotal=spans.reduce((sum,item)=>sum+Math.sqrt(item.span),0)||1;
 const minimumWidth=spans.length?Math.min(86,contentWidth/spans.length):0;
 const flexibleWidth=Math.max(0,contentWidth-minimumWidth*spans.length);
 let contentRight=0;
 const lanes=spans.map(item=>{
  const width=spaces.length===1?contentWidth:minimumWidth+flexibleWidth*Math.sqrt(item.span)/weightTotal;
  const lane={...item.space,span:item.span,x:contentRight,width,scale:width/item.span};
  contentRight+=width;
  return lane;
 });
 const projection=verticalProjection(fn,entries,SOURCE_LINE_HEIGHT*view.scaleY);
 let lastFunctionLabelY=-Infinity;
 const axisLabels=projection.labels.filter(label=>{
  if(label.name){if(label.y-lastFunctionLabelY<18)return false;lastFunctionLabelY=label.y;return true;}
  return view.scaleY>=.5;
 });
 const gridWidth=34*view.scaleX,gridHeight=34*view.scaleY*Math.max(1,Math.ceil(8/(34*view.scaleY)));
 const clampView=next=>({...next,x:clampOffset(next.x,plotWidth,contentRight),y:clampOffset(next.y,plotHeight,projection.height)});

 useEffect(()=>{
  const current=viewRef.current,scaleX=clamp(current.scaleX,1,3),scaleY=clamp(current.scaleY,minScaleY,3);
  const next={...current,scaleX,scaleY,x:clampOffset(current.x,plotWidth,contentRight),y:clampOffset(current.y,plotHeight,projection.height)};
  if(next.x!==current.x||next.y!==current.y||next.scaleX!==current.scaleX||next.scaleY!==current.scaleY)moveView(next);
 },[contentRight,projection.height,plotWidth,plotHeight,minScaleY]);
 useEffect(()=>{moveView({x:0,y:0,scaleX:1,scaleY:1});},[fn.name,spaceName,fitUsed]);
 useEffect(()=>{
  const element=plot.current;
  const wheel=event=>{
   event.preventDefault();event.stopPropagation();
   const rect=element.getBoundingClientRect(),delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?element.clientHeight:1);
   const factor=Math.exp(-Math.max(-200,Math.min(200,delta))*.002);
   moveView(zoomViewport(viewRef.current,factor,{x:event.clientX-rect.left,y:event.clientY-rect.top},limitsRef.current.minScaleY));
  };
  element.addEventListener('wheel',wheel,{passive:false});
  return()=>element.removeEventListener('wheel',wheel);
 },[]);
 useEffect(()=>{
  if(selected===null)return;
  const box=fn.boxes[selected],position=projection.positions.get(box),lane=lanes.find(item=>item.space===box?.space);
  if(!box||!position||!lane)return;
  const address=mapGeometry(fn,lane,box,fitUsed,lane.scale),current=viewRef.current;
  const targetX=lane.x+address.x+address.width/2,targetY=position.y+position.height/2;
  const screenX=current.x+targetX,screenY=current.y+targetY;
  if(screenX<24||screenX>plotWidth-24||screenY<24||screenY>plotHeight-24)moveView(clampView({...current,x:plotWidth/2-targetX,y:plotHeight/2-targetY}));
 },[selected,fn,projection.positions,lanes]);

 const selectKey=(event,index)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();onSelect(`box.${index}`);}};
 const endPan=event=>{pan.current=null;event.currentTarget.style.cursor='grab';if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);};
 const zoomBy=factor=>moveView(zoomViewport(viewRef.current,factor,{x:plotWidth/2,y:plotHeight/2},minScaleY));
 const fitView=()=>moveView({x:0,y:0,scaleX:1,scaleY:minScaleY});

 return <div ref={host} aria-label="Function Memory Map，固定坐标轴，滚轮缩放比例尺，按住空白区域拖动" style={{position:'absolute',inset:'128px 28px 0',overflow:'hidden'}}>
  <svg width={AXIS_WIDTH} height={HEADER_HEIGHT} aria-hidden="true" style={{position:'absolute',left:0,top:0,display:'block',background:'var(--stage)'}}><text x="4" y="20" fill="var(--muted)" fontFamily="var(--font-data)" fontSize="12">function / line</text></svg>
  <svg width={plotWidth} height={HEADER_HEIGHT} role="img" aria-label="固定顶部地址轴" style={{position:'absolute',left:AXIS_WIDTH,top:0,display:'block',background:'var(--stage)'}}>
   <g transform={`translate(${view.x} 0)`}>{lanes.map(lane=><g key={lane.space} data-tooltip={memorySpaceDescription(lane.space)}><text x={lane.x+8} y="18" fill="var(--text)" fontSize="14" fontWeight="600">{memorySpaceLabel(lane.space)}</text><text x={lane.x+8} y="39" fill="var(--faint)" fontFamily="var(--font-data)" fontSize="11">0</text><text x={lane.x+lane.width-8} y="39" textAnchor="end" fill="var(--muted)" fontFamily="var(--font-data)" fontSize="11">{bytes(lane.span)}</text>{Array.from({length:Math.max(0,Math.floor(lane.width/34)-1)},(_,i)=>{const offset=(i+1)*34;const major=(i+1)%Math.max(2,Math.ceil(76/34))===0;return <g key={i}><line x1={lane.x+offset} x2={lane.x+offset} y1={major?42:45} y2={48} stroke="var(--faint)" strokeOpacity=".5"/>{major&&offset>60&&offset<lane.width-65&&<text x={lane.x+offset} y="39" textAnchor="middle" fill="var(--muted)" fontFamily="var(--font-data)" fontSize="11">{bytes(lane.span*offset/lane.width)}</text>}</g>;})}</g>)}</g>
  </svg>
  <svg width={AXIS_WIDTH} height={plotHeight} role="img" aria-label="固定左侧函数与源码行轴" style={{position:'absolute',left:0,top:HEADER_HEIGHT,display:'block',background:'var(--stage)'}}>
   <g transform={`translate(0 ${view.y})`}>{axisLabels.map((label,index)=>label.name?<text key={`f-${index}`} data-tooltip={label.name} x="4" y={label.y} fill="var(--muted)" fontSize="11" fontFamily="var(--font-data)">{truncate(label.name)}</text>:<text key={`l-${index}`} x="100" y={label.y} textAnchor="end" fill="var(--faint)" fontSize="10" fontFamily="var(--font-data)">L{label.line}</text>)}</g>
  </svg>
  <div ref={plot} aria-label="内存占用数据区" style={{position:'absolute',left:AXIS_WIDTH,top:HEADER_HEIGHT,width:plotWidth,height:plotHeight,overflow:'hidden',touchAction:'none',cursor:'grab'}}
   onPointerDown={event=>{if(event.button!==0||event.target.closest?.('[data-box-index]'))return;event.preventDefault();const current=viewRef.current;pan.current={x:event.clientX,y:event.clientY,left:current.x,top:current.y};event.currentTarget.setPointerCapture(event.pointerId);event.currentTarget.style.cursor='grabbing';}}
   onPointerMove={event=>{if(!pan.current)return;const origin=pan.current;moveView(clampView({...viewRef.current,x:origin.left+event.clientX-origin.x,y:origin.top+event.clientY-origin.y}));}}
   onPointerUp={endPan} onPointerCancel={endPan} onLostPointerCapture={event=>{pan.current=null;event.currentTarget.style.cursor='grab';}}>
   <svg width={plotWidth} height={plotHeight} aria-label="Tile 地址与静态生命周期">
    <defs>
     <pattern id="mi-coordinate-grid" width={gridWidth} height={gridHeight} patternUnits="userSpaceOnUse" x={view.x} y={view.y}><path d={`M ${gridWidth} 0 H 0 V ${gridHeight}`} fill="none" stroke="var(--faint)" strokeOpacity=".16" strokeWidth="1"/></pattern>
     {lanes.map(lane=><clipPath id={`mi-lane-${lane.space}`} key={lane.space}><rect x={lane.x} width={lane.width} height={projection.height}/></clipPath>)}
    </defs>
    <rect width={plotWidth} height={plotHeight} fill="url(#mi-coordinate-grid)" pointerEvents="none" aria-label="随横纵比例尺缩放的浅网格"/>
    <g transform={`translate(${view.x} ${view.y})`}>{lanes.map(lane=><g key={lane.space} clipPath={`url(#mi-lane-${lane.space})`}>
     {entries.filter(({box})=>box.space===lane.space).map(({box,index})=>{
      const address=mapGeometry(fn,lane,box,fitUsed,lane.scale),lifetime=projection.positions.get(box),active=selected===index||line!==null&&box.start<=line&&line<=box.end;
      if(!lifetime)return null;
      const anomaly=box.conflict||box.offset+box.size>lane.limit,tooltip=`${box.functionName||fn.name}\n${box.name}${box.aliases.length?' +'+box.aliases.length+' aliases':''}\n${box.base} · [${box.offset}, ${box.offset+box.size}) B\nL${box.start}–L${box.end}${box.view?' · View':''}`;
      const textColor=memoryBlockTextColor(box,colorMode,active||anomaly);
      return <g key={index} data-box-index={index} data-tooltip={tooltip} role="button" tabIndex="0" aria-label={`${box.name} · ${box.space} · ${box.offset}–${box.offset+box.size} B · L${box.start}–L${box.end}${box.view?' · View':''}`} onClick={()=>onSelect(`box.${index}`)} onKeyDown={event=>selectKey(event,index)} cursor="pointer">
       <rect x={lane.x+address.x} y={lifetime.y} width={Math.max(2,address.width)} height={lifetime.height} rx="3" fill={anomaly?'var(--anomaly)':active?'var(--memory)':memoryBlockColor(box,colorMode)} stroke="var(--stage)" strokeWidth="2" paintOrder="stroke fill"/>
       {address.width>44&&lifetime.height>14&&<svg x={lane.x+address.x+4} y={lifetime.y+2} width={Math.max(0,address.width-8)} height={Math.max(0,lifetime.height-4)} overflow="hidden" pointerEvents="none"><text y="12" fill={textColor} fontSize="11" fontFamily="var(--font-data)" fontWeight="600">{box.name}{box.aliases.length?` +${box.aliases.length}`:''}</text>{lifetime.height>30&&<text y="26" fill={textColor} fontSize="10" fontFamily="var(--font-data)">{bytes(box.size)} · L{box.start}–L{box.end}</text>}</svg>}
      </g>})}
    </g>)}</g>
   </svg>
  </div>
  <div className="canvas-tools" data-liquid-glass="control" aria-label="内存图缩放">
   <button type="button" className="zoom-trigger" aria-label="展开缩放控件"><Icon name="zoom"/></button>
   <div className="zoom-options">
    <button type="button" onClick={()=>zoomBy(.8)} aria-label="缩小视图" data-tooltip="缩小">−</button>
    <output data-liquid-readout aria-label="当前横纵比例">X {Math.round(view.scaleX*100)}% · Y {Math.round(view.scaleY*100)}%</output>
    <button type="button" onClick={()=>zoomBy(1.25)} aria-label="放大视图" data-tooltip="放大">＋</button>
    <span className="tool-divider"/>
    <button type="button" onClick={fitView} aria-label="适配完整画布" data-tooltip="适配完整画布"><Icon name="fit"/></button>
    <button type="button" onClick={()=>moveView({x:0,y:0,scaleX:1,scaleY:1})} aria-label="恢复默认比例" data-tooltip="恢复默认比例"><Icon name="reset"/></button>
   </div>
  </div>
 </div>;
}
