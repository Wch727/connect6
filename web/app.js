'use strict';
const $ = id => document.getElementById(id);
let state = null, pending = false, connected = true, moveSignature = '', mutation = 0;
const cells = [];
const svg = document.querySelector('.grid-lines');
const ns = 'http://www.w3.org/2000/svg';
for (let i=0;i<19;i++) {
  for (const vertical of [false,true]) {
    const line=document.createElementNS(ns,'line');
    Object.entries(vertical ? {x1:i*40,y1:0,x2:i*40,y2:720} : {x1:0,y1:i*40,x2:720,y2:i*40}).forEach(([k,v])=>line.setAttribute(k,v));
    svg.append(line);
  }
  const x=document.createElement('span');x.textContent=String.fromCharCode(65+i);x.style.left=(i/18*100)+'%';document.querySelector('.coordinates.top').append(x);
  const y=document.createElement('span');y.textContent=19-i;y.style.top=(i/18*100)+'%';document.querySelector('.coordinates.left').append(y);
}
for (const r of [3,9,15]) for(const c of [3,9,15]) {
  const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',c*40);dot.setAttribute('cy',r*40);dot.setAttribute('r',3);svg.append(dot);
}
for(let a=0;a<361;a++) {
  const button=document.createElement('button');button.className='point';button.type='button';button.tabIndex=a===180?0:-1;
  button.dataset.action=a;button.setAttribute('aria-label',coordinate(a));
  button.addEventListener('click',()=>mutate('/api/move',{action:a}));
  button.addEventListener('keydown',e=>{
    const offsets={ArrowLeft:-1,ArrowRight:1,ArrowUp:-19,ArrowDown:19};
    if(!(e.key in offsets))return;e.preventDefault();
    const next=a+offsets[e.key];if(next<0||next>=361)return;
    if((e.key==='ArrowLeft'||e.key==='ArrowRight')&&Math.floor(next/19)!==Math.floor(a/19))return;
    cells.forEach(b=>b.tabIndex=-1);cells[next].tabIndex=0;cells[next].focus();
  });
  cells.push(button);$('intersections').append(button);
}
function coordinate(a){return String.fromCharCode(65+a%19)+(19-Math.floor(a/19));}
function clock(seconds){const s=Math.max(0,Math.ceil(seconds));return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');}
function notice(message){$('notice').textContent=message;$('notice').hidden=!message;}
async function request(path,data){
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'操作失败，请重试。');return result;
}
function render(s){
 state=s;
 $('book-hint').textContent=s.book.message || `已载入 ${s.book.entries.toLocaleString()} 个局面，未命中时由 AI 搜索`;
 const canPlay=connected&&!pending&&s.active&&!s.over&&!s.error&&!s.thinking&&s.player===s.config.human;
 const last=s.moves.at(-1);
 for(let a=0;a<361;a++){
   const r=Math.floor(a/19),c=a%19,p=s.board[r][c],b=cells[a];
   // Keep focusable cells available for arrow navigation; enforce legality in click handling.
   b.setAttribute('aria-disabled',String(!canPlay||p!==0));
   b.setAttribute('aria-label',coordinate(a)+(p===1?'，黑子':p===-1?'，白子':'，空位'));
   b.classList.toggle('last',!!last&&last.row===r&&last.col===c);
   if(b.dataset.piece!==String(p)){
     b.replaceChildren();if(p){const stone=document.createElement('span');stone.className='piece '+(p===1?'black':'white');b.append(stone);}b.dataset.piece=p;
   }
 }
 $('move-count').textContent=s.moves.length+' 子';
 $('black-time').textContent=clock(s.clock.black);$('white-time').textContent=clock(s.clock.white);
 for(const [name,p] of [['black',1],['white',-1]]){
  $(name+'-player').classList.toggle('active',s.active&&!s.over&&s.player===p);
  $(name+'-name').textContent=s.config.human===p?'你':s.config.model==='heuristic'?'启发式 AI':s.config.model;
 }
 $('undo').disabled=pending||!connected||!s.can_undo;
 $('export').disabled=pending||!connected||!s.active;
 $('start').textContent=s.active?'重新开始 →':'开始对局 →';
 $('status').classList.toggle('thinking',s.thinking&&!s.over&&!s.error);
 let status='设置好对局，开始第一手。';
 if(s.error)status='对局已暂停';
 else if(s.over)status=(s.winner===null?'和棋':s.winner===s.config.human?'你赢了':'AI 获胜')+(s.reason==='timeout'?` · ${s.winner===1?'白方':'黑方'}超时`:'');
 else if(s.active)status=s.thinking?'AI 思考中…':`轮到你 · ${s.player===1?'黑方':'白方'}还需落 ${s.remaining_stones} 子`;
 $('status').textContent=status;
 if(s.error)notice(s.error);
 const signature=s.generation+':'+s.moves.length;
 if(signature!==moveSignature){
  moveSignature=signature;
  if(!s.moves.length)$('records').innerHTML='<p class="empty">棋盘还很安静。<br>开始对局后，落子会记录在这里。</p>';
  else {
   $('records').replaceChildren();
   for(let i=0;i<s.moves.length;){
    const row=document.createElement('div');row.className='record';const label=document.createElement('small');label.textContent=i===0?'首手':String(Math.floor((i+1)/2)+1).padStart(2,'0');row.append(label);
    const count=i===0?1:Math.min(2,s.moves.length-i);
    for(let j=0;j<count;j++){
     const m=s.moves[i+j],span=document.createElement('span'),dot=document.createElement('i');dot.className=m.player===1?'b':'w';span.append(dot,document.createTextNode(coordinate(m.row*19+m.col)));if(m.source==='book'){const tag=document.createElement('small');tag.textContent=' · 库';span.append(tag);}row.append(span);
    }
    $('records').append(row);i+=count;
   }
   $('records').scrollTop=$('records').scrollHeight;
  }
 }
}
async function mutate(path,data){
 if(pending)return;
 if(path==='/api/move'&&(!state||!connected||!state.active||state.over||state.thinking||state.error||state.player!==state.config.human||state.board[Math.floor(data.action/19)][data.action%19]))return;
 pending=true;mutation++;notice('');$('start').disabled=true;if(state)render(state);
 try{const s=await request(path,data);connected=true;notice('');render(s);}catch(e){notice(e.message);}
 finally{pending=false;$('start').disabled=false;if(state)render(state);}
}
$('settings').addEventListener('submit',e=>{
 e.preventDefault();
 if(state?.active&&state.moves.length&&!state.over&&!window.confirm('开始新一局？当前对局会被清空，可先导出棋谱。'))return;
 mutate('/api/new',{human:Number(new FormData(e.target).get('human')),model:$('model').value,simulations:Number($('simulations').value),seconds:Number($('minutes').value)*60,device:$('device').value,rules:$('rules').checked,book:$('book').checked});
});
$('undo').addEventListener('click',()=>mutate('/api/undo',{}));
$('export').addEventListener('click',async()=>{
 try{const record=await request('/api/export');const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='connect6-game.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){notice(e.message);}
});
function modelHint(){const neural=$('model').value!=='heuristic';$('simulations').disabled=!neural;$('device').disabled=!neural;$('rules').disabled=!neural;$('model-hint').textContent=neural?'将按所选文件加载模型。首次加载可能需要几秒钟。':'无需权重，使用两子战术与有限候选搜索。搜索次数和设备选项用于神经网络模型。';}
$('model').addEventListener('change',modelHint);
async function poll(){
 try{if(!pending){const version=mutation,s=await request('/api/state');if(version!==mutation||pending)return;if(!connected)notice('');connected=true;render(s);}}
 catch(e){connected=false;notice('本地服务连接中断。请确认 Python 服务仍在运行，恢复后页面会自动重连。');if(state)render(state);}
 finally{setTimeout(poll,500);}
}
request('/api/models').then(data=>{for(const name of data.models){const o=document.createElement('option');o.value=name;o.textContent=name;$('model').append(o);}}).catch(()=>notice('模型列表读取失败，请检查本地服务。'));
modelHint();poll();
