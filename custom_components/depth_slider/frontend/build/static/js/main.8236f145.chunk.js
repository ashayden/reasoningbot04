(this["webpackJsonpdepth-slider"]=this["webpackJsonpdepth-slider"]||[]).push([[0],{14:function(n,e,t){"use strict";t.r(e);var a,s,i,c,o,r,p,d=t(2),l=t(0),b=t.n(l),j=t(10),h=t.n(j),f=t(3),u=t(7),x=t(1);const O=f.a.div(a||(a=Object(d.a)(["\n  width: 100%;\n  padding: 20px 0;\n  position: relative;\n"]))),g=f.a.div(s||(s=Object(d.a)(['\n  font-family: "Source Sans Pro", sans-serif;\n  font-size: 14px;\n  color: #fff;\n  margin-bottom: 8px;\n']))),v=f.a.div(i||(i=Object(d.a)(["\n  width: 100%;\n  height: 2px;\n  background: rgba(255, 255, 255, 0.2);\n  position: relative;\n"]))),m=f.a.div(c||(c=Object(d.a)(["\n  height: 100%;\n  background: #0066cc;\n  width: ","%;\n  position: absolute;\n  left: 0;\n  top: 0;\n"])),(n=>n.percentage)),w=f.a.div(o||(o=Object(d.a)(["\n  width: 24px;\n  height: 24px;\n  background: white;\n  border-radius: 50%;\n  position: absolute;\n  top: 50%;\n  left: ","%;\n  transform: translate(-50%, -50%);\n  cursor: pointer;\n  transition: transform 0.1s ease;\n\n  &:hover {\n    transform: translate(-50%, -50%) scale(1.1);\n  }\n"])),(n=>n.percentage)),k=f.a.div(r||(r=Object(d.a)(["\n  display: flex;\n  justify-content: space-between;\n  margin-top: 10px;\n  position: relative;\n  padding: 0 12px;\n"]))),y=f.a.span(p||(p=Object(d.a)(["\n  color: ",';\n  font-family: "Source Sans Pro", sans-serif;\n  font-size: 14px;\n  cursor: pointer;\n  transition: color 0.2s ease;\n\n  &:hover {\n    color: #0066cc;\n  }\n'])),(n=>n.active?"#0066cc":"#fff")),S=Object(u.b)((n=>{let{args:e,disabled:t}=n;const{options:a=[],value:s="",label:i="Analysis Depth"}=e,[c,o]=Object(l.useState)(s),r=a.indexOf(c)/Math.max(a.length-1,1)*100;Object(l.useEffect)((()=>{u.a.setFrameHeight()}),[]),Object(l.useEffect)((()=>{s!==c&&o(s)}),[s]);return Object(x.jsxs)(O,{children:[Object(x.jsx)(g,{children:i}),Object(x.jsxs)(v,{children:[Object(x.jsx)(m,{percentage:r}),Object(x.jsx)(w,{percentage:r})]}),Object(x.jsx)(k,{children:a.map((n=>Object(x.jsx)(y,{active:n===c,onClick:()=>(n=>{t||(o(n),u.a.setComponentValue(n))})(n),children:n},n)))})]})}));h.a.render(Object(x.jsx)(b.a.StrictMode,{children:Object(x.jsx)(S,{})}),document.getElementById("root"))}},[[14,1,2]]]);