import React from 'react';
import ReactDOM from 'react-dom/client';
import CaseApp from './CaseApp';
import MemoryInspectorApp from './MemoryInspectorApp';
function Router(){
 const readCase=()=>new URLSearchParams(location.hash.slice(1)).get('case');
 const [caseId,setCaseId]=React.useState(readCase);
 React.useEffect(()=>{const update=()=>setCaseId(readCase());window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);},[]);
 return caseId==='MI-01'?<MemoryInspectorApp/>:<CaseApp/>;
}
ReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode><Router /></React.StrictMode>);
