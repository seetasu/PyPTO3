export default function Icon({name}){
 const paths={
  settings:<><path d="M4 7h5m4 0h7M4 17h9m4 0h3"/><circle cx="11" cy="7" r="2"/><circle cx="15" cy="17" r="2"/></>,
  fit:<><path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"/><path d="M3 8l5-5M21 8l-5-5M3 16l5 5M21 16l-5 5"/></>,
  reset:<><path d="M4 12a8 8 0 1 0 2.34-5.66L4 8.68"/><path d="M4 4v4.68h4.68"/></>,
  info:<><circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><circle cx="12" cy="8" r=".7" fill="currentColor" stroke="none"/></>,
  detail:<><path d="M4 6h16M4 12h16M4 18h10"/><circle cx="18" cy="18" r="2"/></>,
  close:<path d="M6 6l12 12M18 6L6 18"/>,
  play:<path d="M8 5l11 7-11 7Z"/>,
  pause:<path d="M8 5v14M16 5v14"/>,
  zoom:<><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6M10 7v6"/></>,
 };
 return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}
