import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import ContextDiagram from './ContextDiagram';
import contextData from '../data/csa-context.json';
import './csa-context.css';

export default function CsaContext({onClose}) {
  const [width,setWidth]=useState(640),[zoom,setZoom]=useState(1),[selected,setSelected]=useState(null);
  const [expanded,setExpanded]=useState(0),[offset,setOffset]=useState({x:0,y:0});
  const offsetRef=useRef({x:0,y:0});
  const moveTo=value=>{offsetRef.current=value;setOffset(value);};
  const graph=useMemo(()=>{
    const variant=contextData.variants[expanded];
    return {...variant,...Object.fromEntries(['nodes','edges','clusters'].map(kind=>[kind,variant[kind].map(index=>contextData.pool[kind][index])]))};
  },[expanded]);
  const toggle=id=>{setExpanded(mask=>mask^(1<<contextData.foldable.indexOf(id)));setSelected(null);};
  const panel=useRef(null),focus=useRef(null),drag=useRef(null),viewport=useRef(null),pan=useRef(null),zoomRef=useRef(1);
  useLayoutEffect(()=>{
    const el=viewport.current,padding=16;
    const scale=Math.max(1,el.clientHeight-padding*2)/graph.height;
    const initialZoom=scale*graph.width/Math.max(width-34,1000);
    zoomRef.current=initialZoom;setZoom(initialZoom);
    moveTo({x:(el.clientWidth-graph.width*scale)/2,y:(el.clientHeight-graph.height*scale)/2});
    // Initial framing only: later interactions retain the user's camera.
  },[]);
  const changeZoom=(value,point)=>{
    const el=viewport.current,previous=zoomRef.current,next=Math.max(.01,Math.min(3,value));
    if(!el||next===previous)return;
    const x=point?.x??el.clientWidth/2,y=point?.y??el.clientHeight/2;
    const previousOffset=offsetRef.current;
    const nextOffset={x:x-(x-previousOffset.x)*next/previous,y:y-(y-previousOffset.y)*next/previous};
    zoomRef.current=next;
    flushSync(()=>{setZoom(next);moveTo(nextOffset);});
  };
  useEffect(()=>{
    const el=viewport.current;
    const wheel=e=>{
      e.preventDefault();e.stopPropagation();
      const rect=el.getBoundingClientRect(),delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?el.clientHeight:1);
      changeZoom(zoomRef.current*Math.exp(-Math.max(-200,Math.min(200,delta))*.002),{x:e.clientX-rect.left,y:e.clientY-rect.top});
    };
    el.addEventListener('wheel',wheel,{passive:false});
    return()=>el.removeEventListener('wheel',wheel);
  },[]);
  const endPan=e=>{
    pan.current=null;e.currentTarget.classList.remove('is-panning');
    if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);
  };
  useEffect(()=>{panel.current?.focus();const key=e=>{if(e.key==='Escape')onClose();};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[onClose]);
  const model=useMemo(()=>{
    const points=graph.nodes.filter(n=>n.focus);
    const x=Math.min(...points.map(p=>p.x-p.width/2))-8,y=Math.min(...points.map(p=>p.y-p.height/2))-18;
    return {bounds:{x,y,width:Math.max(...points.map(p=>p.x+p.width/2))+8-x,height:Math.max(...points.map(p=>p.y+p.height/2))+10-y}};
  },[graph]);
  const cap=w=>Math.max(320,Math.min(window.innerWidth-24,w));
  const locate=()=>{
    const el=viewport.current,scale=Math.max(width-34,1000)*zoomRef.current/graph.width;
    moveTo({x:el.clientWidth/2-(model.bounds.x+model.bounds.width/2)*scale,y:el.clientHeight/2-(model.bounds.y+model.bounds.height/2)*scale});
  };
  return <><div className="csa-context-dismiss" onClick={onClose}/><aside ref={panel} tabIndex={-1} className="csa-context" role="dialog" aria-label="CSA 原始计算图" style={{width}}>
    <header><h2 title="来源：CSA workbench。展示投影、压缩、检索和输出步骤；白框标出 qk_pv 对应的 QK、块统计与 PV。">CSA 原始计算图</h2>
      <nav aria-label="计算图操作"><button title="定位 QK → 块统计 → PV 核心" onClick={locate}>qk_pv</button><button onClick={()=>changeZoom(zoomRef.current-.25)} aria-label="缩小计算图">−</button><span>{Math.round(zoom*100)}%</span><button onClick={()=>changeZoom(zoomRef.current+.25)} aria-label="放大计算图">+</button></nav>
      <button className="context-close" onClick={onClose} aria-label="关闭计算图">×</button>
    </header>
    <div ref={viewport} className="context-scroll" aria-label="计算图画布，滚轮缩放，按住空白区域拖拽移动"
      onPointerDown={e=>{
        if(e.button!==0||e.target.closest('.context-node'))return;
        e.preventDefault();const el=e.currentTarget;
        pan.current={x:e.clientX,y:e.clientY,left:offsetRef.current.x,top:offsetRef.current.y};
        el.setPointerCapture(e.pointerId);el.classList.add('is-panning');
      }}
      onPointerMove={e=>{if(!pan.current)return;const p=pan.current;moveTo({x:p.left+e.clientX-p.x,y:p.top+e.clientY-p.y});}}
      onPointerUp={endPan} onPointerCancel={endPan} onLostPointerCapture={()=>{pan.current=null;viewport.current?.classList.remove('is-panning');}}>
      <div className="context-scene" style={{transform:`translate(${offset.x}px, ${offset.y}px)`}}><ContextDiagram graph={graph} width={Math.max(width-34,1000)*zoom} bounds={model.bounds} focusRef={focus} selected={selected} onSelect={n=>n.foldable?toggle(n.id):setSelected(n)} onToggle={toggle}/></div>
    </div>
    {selected&&<footer><div><strong>{selected.label}</strong><p title={selected.detail}>{selected.detail}</p></div><button onClick={()=>setSelected(null)} aria-label="收起节点详情">×</button></footer>}
    <div className="context-resize" role="separator" tabIndex="0" aria-label="调整计算图宽度" aria-orientation="vertical" aria-valuenow={width} aria-valuemin="320" aria-valuemax={window.innerWidth-24} onPointerDown={e=>{drag.current={x:e.clientX,width};e.currentTarget.setPointerCapture(e.pointerId);}} onPointerMove={e=>{if(drag.current)setWidth(cap(drag.current.width+e.clientX-drag.current.x));}} onPointerUp={()=>drag.current=null} onPointerCancel={()=>drag.current=null} onKeyDown={e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();setWidth(w=>cap(w+(e.key==='ArrowRight'?32:-32)));}}}/>
  </aside></>;
}
