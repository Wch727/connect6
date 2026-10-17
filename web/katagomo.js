'use strict';
const $=id=>document.getElementById(id),COLS='ABCDEFGHJKLMNOPQRST';
let state=null,pending=false,mutation=0,recordSignature='',connected=true;
const cells=[],svg=document.querySelector('.grid-lines'),ns='http://www.w3.org/2000/svg';
const coordinate=a=>COLS[a%19]+(19-Math.floor(a/19));
function notice(text){$('notice').textContent=text;$('notice').hidden=!text;}
for(let i=0;i<19;i++){
  for(const vertical of [false,true]){
    const line=document.createElementNS(ns,'line');
    for(const [key,value] of Object.entries(vertical?{x1:i*40,y1:0,x2:i*40,y2:720}:{x1:0,y1:i*40,x2:720,y2:i*40}))line.setAttribute(key,value);
    svg.append(line);
  }
  const x=document.createElement('span');x.textContent=COLS[i];x.style.left=(i/18*100)+'%';document.querySelector('.coordinates.top').append(x);
  const y=document.createElement('span');y.textContent=19-i;y.style.top=(i/18*100)+'%';document.querySelector('.coordinates.left').append(y);
}
for(const r of [3,9,15])for(const c of [3,9,15]){
  const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',c*40);dot.setAttribute('cy',r*40);dot.setAttribute('r',3);svg.append(dot);
}
for(let a=0;a<361;a++){
  const b=document.createElement('button');b.type='button';b.className='point';b.dataset.action=a;b.tabIndex=a===180?0:-1;
  b.setAttribute('aria-label',coordinate(a));b.addEventListener('click',()=>{
    if(!state||!canPlay(state)||state.board[Math.floor(a/19)][a%19])return;
    mutate('/api/move',{action:a});
  });
  b.addEventListener('keydown',e=>{
    const offsets={ArrowLeft:-1,ArrowRight:1,ArrowUp:-19,ArrowDown:19};if(!(e.key in offsets))return;e.preventDefault();
    const next=a+offsets[e.key];if(next<0||next>=361)return;
    if(['ArrowLeft','ArrowRight'].includes(e.key)&&Math.floor(next/19)!==Math.floor(a/19))return;
    cells.forEach(cell=>cell.tabIndex=-1);cells[next].tabIndex=0;cells[next].focus();
  });
  cells.push(b);$('intersections').append(b);
}
function canPlay(s){return connected&&!pending&&s.active&&!s.busy&&!s.error&&!s.over&&(s.mode==='review'||s.player===s.human);}
async function request(path,data){
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const value=await response.json();if(!response.ok)throw Error(value.error||'操作失败，请重试。');return value;
}
function settings(){const data=new FormData($('settings'));return {human:Number(data.get('human')),mode:data.get('mode'),seconds:Number($('seconds').value)};}
function render(s){
  state=s;const playable=canPlay(s),shown=s.moves.slice(0,s.cursor),last=shown.at(-1),numbers=new Map(shown.map((m,i)=>[m.action,i+1]));
  const hints=new Map(s.hints.map((a,i)=>[a,i+1]));
  for(let a=0;a<361;a++){
    const p=s.board[Math.floor(a/19)][a%19],b=cells[a],number=$('numbers').checked?numbers.get(a):'',hint=hints.get(a);
    b.setAttribute('aria-disabled',String(!playable||!!p));b.setAttribute('aria-label',coordinate(a)+(p===1?'，黑子':p===-1?'，白子':'，空位')+(hint?'，推荐第'+hint+'颗':''));
    b.classList.toggle('last',last?.action===a);
    const key=[p,number,hint].join(':');
    if(b.dataset.content!==key){
      b.replaceChildren();if(p){const stone=document.createElement('span');stone.className='piece '+(p===1?'black':'white');stone.textContent=number||'';b.append(stone);}
      if(hint&&!p){const mark=document.createElement('span');mark.className='hint';mark.textContent=hint;b.append(mark);}b.dataset.content=key;
    }
  }
  const review=s.mode==='review';$('heading').textContent=review?'摆棋与复盘':'与 Katagomo 对弈';$('mode-caption').textContent='十九路 · '+(review?'复盘':'对弈');
  $('engine-status').textContent=s.engine_available?'Katagomo · 本地':'未找到 Katagomo';
  for(const [name,color] of [['black',1],['white',-1]]){
    $(name+'-player').classList.toggle('active',s.active&&!s.over&&s.player===color);
    $(name+'-name').textContent=review?'棋谱':s.human===color?'你':'Katagomo';
    $(name+'-turn').textContent=s.active&&!s.over&&s.player===color?'还需 '+s.remaining_stones+' 子':'';
  }
  let status='选择执子，开始对局。';
  if(s.error)status='本次思考未完成';
  else if(s.busy)status=s.job==='hint'?'正在寻找推荐落点…':'Katagomo 思考中…';
  else if(s.over)status=(s.winner===null?'和棋':s.winner===1?'黑方获胜':'白方获胜')+(review?' · 可以回退复盘':'');
  else if(s.active)status=review?`轮到${s.player===1?'黑方':'白方'} · 可落 ${s.remaining_stones} 子`:`轮到你 · 还需落 ${s.remaining_stones} 子`;
  $('status').textContent=status;$('status').classList.toggle('thinking',s.busy);
  $('undo').disabled=pending||!connected||!s.can_undo;
  $('analyze').disabled=pending||!connected||!s.active||s.busy||s.over||!s.engine_available;
  $('apply-hint').hidden=!s.hints.length;$('apply-hint').disabled=pending||s.busy;
  $('hint-text').textContent=s.hints.length?'推荐：'+s.hints.map(coordinate).join(' → '):'推荐只作预览，确认后再落子。';
  $('seconds').disabled=pending||s.busy;
  if(document.activeElement!==$('seconds')&&!pending)$('seconds').value=s.seconds;
  $('start').disabled=pending||!connected;
  $('start').textContent=review?'清空棋盘 →':s.active?'重新开始 →':'开始对局 →';
  for(const input of document.querySelectorAll('input[name="mode"]'))input.checked=input.value===s.mode;
  $('first').disabled=$('previous').disabled=pending||!connected||s.cursor===0;
  $('last').disabled=$('next').disabled=pending||!connected||s.cursor===s.total;
  $('cursor').disabled=pending||!connected||s.total===0;$('cursor').max=s.total;$('cursor').value=s.cursor;
  $('progress').textContent=s.cursor+' / '+s.total+' 子';$('record-label').textContent=s.total?'点击落点跳转':'还没有落子';
  $('export').disabled=pending||!connected||!s.total;
  $('review-note').textContent=review?'点击棋盘，可从当前局面续写棋谱':'首次启动引擎需要几秒钟';
  if(s.error)notice(s.error);
  const signature=s.version+':'+s.cursor+':'+s.total;
  if(signature!==recordSignature){
    recordSignature=signature;$('records').replaceChildren();
    if(!s.total){const p=document.createElement('p');p.className='empty';p.textContent='导入棋谱，逐步复盘。也可以直接开始一局。';$('records').append(p);}
    for(let i=0;i<s.total;){
      const row=document.createElement('div');row.className='record';const label=document.createElement('small');label.textContent=i===0?'首手':String(Math.floor((i+1)/2)+1).padStart(2,'0');row.append(label);
      const count=i===0?1:Math.min(2,s.total-i);
      for(let j=0;j<count;j++){
        const index=i+j,m=s.moves[index],button=document.createElement('button'),dot=document.createElement('i');button.type='button';
        dot.className=m.player===1?'b':'w';button.append(dot,document.createTextNode(m.coordinate));button.classList.toggle('future',index>=s.cursor);button.classList.toggle('current',index===s.cursor-1);
        button.title='第 '+(index+1)+' 颗';button.addEventListener('click',()=>mutate('/api/seek',{index:index+1}));row.append(button);
      }
      $('records').append(row);i+=count;
    }
    $('records').querySelector('.current')?.scrollIntoView({block:'nearest'});
  }
}
async function mutate(path,data={}){
  if(pending)return;pending=true;mutation++;notice('');if(state)render(state);
  try{const s=await request(path,data);connected=true;render(s);}catch(e){notice(e.message);}
  finally{pending=false;if(state)render(state);}
}
$('settings').addEventListener('submit',e=>{
  e.preventDefault();if(state?.total&&!window.confirm('清空当前棋谱？需要保留的话，请先保存。'))return;
  mutate('/api/new',settings());
});
for(const input of document.querySelectorAll('input[name="mode"]'))input.addEventListener('change',()=>{
  if(!input.checked)return;const data=settings();
  if(!state?.active)mutate('/api/new',data);
  else if(data.mode==='review')mutate('/api/seek',{index:state.cursor});
  else mutate('/api/resume',data);
});
$('seconds').addEventListener('change',()=>{if($('seconds').checkValidity())mutate('/api/settings',{seconds:Number($('seconds').value)});else $('seconds').reportValidity();});
$('numbers').addEventListener('change',()=>state&&render(state));
$('undo').addEventListener('click',()=>mutate('/api/undo'));
$('analyze').addEventListener('click',()=>mutate('/api/analyze'));
$('apply-hint').addEventListener('click',()=>mutate('/api/apply-hint'));
for(const [id,index] of [['first',()=>0],['previous',()=>state.cursor-1],['next',()=>state.cursor+1],['last',()=>state.total]])$(id).addEventListener('click',()=>mutate('/api/seek',{index:index()}));
$('cursor').addEventListener('change',()=>mutate('/api/seek',{index:Number($('cursor').value)}));
$('import-file').addEventListener('change',async e=>{
  const file=e.target.files[0];if(!file)return;
  try{
    if(file.size>524288)throw Error('棋谱超过 512 KB，请选择较小的 SGF 或 JSON 文件。');
    const bytes=await file.arrayBuffer();let text;
    try{text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{text=new TextDecoder('gb18030').decode(bytes);}
    await mutate('/api/import',{text});
  }catch(error){notice(error.message);}finally{e.target.value='';}
});
$('export').addEventListener('click',async()=>{
  try{
    const sgf=$('export-format').value==='sgf',data=await request(sgf?'/api/export-sgf':'/api/export');
    const url=URL.createObjectURL(new Blob([sgf?data.sgf:JSON.stringify(data,null,2)],{type:sgf?'application/x-go-sgf':'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='connect6.'+(sgf?'sgf':'json');a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){notice(e.message);}
});
async function poll(){
  try{
    if(!pending){const version=mutation,s=await request('/api/state');if(pending||version!==mutation)return;
      if(!connected)notice('');connected=true;render(s);}
  }catch(e){connected=false;notice('本地服务连接中断。请确认 Katagomo 界面的 Python 服务仍在运行。');if(state)render(state);}
  finally{setTimeout(poll,350);}
}
poll();
