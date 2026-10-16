import React,{useEffect,useRef,useState} from 'react';
import {samples,sample,sampleHash} from './sample-registry.js';
export function CallableSelector(){
 const [open,setOpen]=useState(false),ref=useRef(null);
 useEffect(()=>{const close=event=>{if(!ref.current?.contains(event.target))setOpen(false);};const key=event=>{if(event.key==='Escape')setOpen(false);};document.addEventListener('pointerdown',close);document.addEventListener('keydown',key);return()=>{document.removeEventListener('pointerdown',close);document.removeEventListener('keydown',key);};},[]);
 return <div className="callable-selector" ref={ref}>
  <button type="button" className="callable-selector-trigger" data-liquid-glass="control" aria-label="选择 callable" aria-haspopup="listbox" aria-expanded={open} onClick={()=>setOpen(value=>!value)}><span>{sample.label}</span><span aria-hidden="true">⌄</span></button>
  {open&&<div className="callable-selector-menu" role="listbox" aria-label="Callable 样本">{samples.map(item=><button type="button" key={item.id} role="option" aria-selected={item.id===sample.id} onClick={()=>{setOpen(false);if(item.id!==sample.id){location.hash=sampleHash(item.id,location.hash);location.reload();}}}><span>{item.label}</span><small>{item.description}</small></button>)}</div>}
 </div>;
}
