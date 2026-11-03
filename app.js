/* Content and timings are grounded in the supplied RapidDOC manuscript. */
(() => {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.documentElement.classList.add('js-ready');
  const revealObserver = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) { entry.target.classList.add('visible'); revealObserver.unobserve(entry.target); }
  }), { threshold: 0.07 });
  document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

  const menuButton = document.getElementById('menu-toggle');
  const mobileMenu = document.getElementById('mobile-nav');
  function closeMenu() { mobileMenu.hidden = true; menuButton.setAttribute('aria-expanded','false'); menuButton.setAttribute('aria-label','Open section navigation'); }
  menuButton.addEventListener('click', () => {
    const open = mobileMenu.hidden; mobileMenu.hidden = !open;
    menuButton.setAttribute('aria-expanded',String(open));
    menuButton.setAttribute('aria-label',open?'Close section navigation':'Open section navigation');
  });
  mobileMenu.querySelectorAll('a').forEach(link => link.addEventListener('click',closeMenu));
  document.addEventListener('keydown',event => { if(event.key==='Escape')closeMenu(); });
  window.addEventListener('resize',() => { if(innerWidth>760)closeMenu(); });

  const figureDialog=document.getElementById('figure-dialog');
  const expandedImage=document.getElementById('figure-dialog-image');
  function showFigure(source,caption,alt) {
    expandedImage.src=source; expandedImage.alt=alt||caption;
    document.getElementById('figure-dialog-caption').textContent=caption;
    figureDialog.showModal();
  }
  document.getElementById('close-figure').addEventListener('click',()=>figureDialog.close());
  figureDialog.addEventListener('click',event=>{if(event.target===figureDialog)figureDialog.close();});
  document.querySelectorAll('[data-figure]').forEach(button=>button.addEventListener('click',()=>showFigure(button.dataset.figure,button.dataset.caption)));
  document.getElementById('learning-full').addEventListener('click',event=>{
    event.preventDefault();showFigure(event.currentTarget.href,document.getElementById('learning-caption').textContent,document.getElementById('learning-image').alt);
  });

  // Keyboard interaction shared by the accessible tab groups.
  document.querySelectorAll('[role="tablist"]').forEach(group => {
    group.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key)) return;
      const tabs = [...group.querySelectorAll('[role="tab"]')];
      let index = tabs.indexOf(document.activeElement);
      if (index < 0) return;
      event.preventDefault();
      if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = tabs.length - 1;
      else index = (index + (['ArrowRight','ArrowDown'].includes(event.key) ? 1 : -1) + tabs.length) % tabs.length;
      tabs[index].focus(); tabs[index].click();
    });
  });
  const selectTab = (button, group) => {
    group.querySelectorAll('[role="tab"]').forEach(tab => {
      const selected = tab === button; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
    });
  };

  const method = [
    {title:'Useful structure, already computed.',equation:'𝒮<sub>F</sub>(v) := CachedSolve(𝒞<sub>F</sub>, v)',detail:'Retain the forward factors and expose their solve action. A fresh target-system factorization is not required.'},
    {title:'A candidate, certified by the target.',equation:'a<sub>0</sub> = 𝒮<sub>F</sub>(g) &nbsp; · &nbsp; r<sub>0</sub> = g − K<sub>B</sub><sup>⊤</sup>a<sub>0</sub>',detail:'Evaluate the candidate with the prescribed target operator. Accept it if its relative residual meets the requested tolerance.'},
    {title:'The same cache. A target-system solve.',equation:'K<sub>B</sub><sup>⊤</sup>𝒮<sub>F</sub>(y) = r<sub>0</sub>',detail:'Use fixed right-preconditioned GMRES when needed, then form a = a₀ + 𝒮F(y). The returned adjoint is checked against the target residual.'}
  ];
  document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => {
    const index = Number(button.dataset.step); const item = method[index];
    selectTab(button, button.closest('[role="tablist"]'));
    document.querySelectorAll('[data-step]').forEach(btn => btn.classList.toggle('active', btn === button));
    const panel = document.getElementById('method-panel'); panel.dataset.stage = index; panel.setAttribute('aria-labelledby', button.id);
    document.getElementById('method-node-title').textContent = item.title;
    document.getElementById('method-equation').innerHTML = item.equation;
    document.getElementById('method-detail').textContent = item.detail;
    document.querySelectorAll('.method-progress i').forEach((el,i) => el.classList.toggle('on', i === index));
  }));

  const tasks = [
    {name:'CartPole',values:[.054896875,.1871749955,53.8169],std:[.0227363072,.0466382608,6.75500085],labels:['0.055','0.19','53.8']},
    {name:'Quadrotor',values:[.40286875,.7402499978,4018.55629375],std:[.4905311298,.1999350864,43.04043224],labels:['0.40','0.74','4018.6']},
    {name:'Robot Arm',values:[.087196875,.1588468731,102.301715625],std:[.055206495,.0547722731,8.761951194],labels:['0.087','0.16','102.3']}
  ];
  const colors = ['#80aaff','#73d4b2','#a0a2af'];
  const names = ['RapidDOC','Refactor','No reuse'];
  const ns = 'http://www.w3.org/2000/svg';
  const svgEl = (name, attributes, text) => {
    const el = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key,value]) => el.setAttribute(key,value));
    if (text !== undefined) el.textContent = text;
    return el;
  };
  const mapY = value => 286 - (Math.log10(Math.max(value,.01)) + 2) / 6.2 * 230;
  function drawChart(index, animate = true) {
    const task = tasks[index]; const container = document.getElementById('ablation-chart');
    const svg = svgEl('svg',{viewBox:'0 0 850 350',role:'img','aria-label':`${task.name} backward time, logarithmic scale: RapidDOC ${task.labels[0]} milliseconds, Refactor ${task.labels[1]}, No reuse ${task.labels[2]}.`});
    svg.append(svgEl('text',{x:33,y:20,fill:'#8b9bb5','font-size':12},'ms · log scale'));
    for (let exp = -2; exp <= 4; exp++) {
      const y = mapY(10 ** exp);
      svg.append(svgEl('line',{x1:65,x2:820,y1:y,y2:y,stroke:'#262b34','stroke-width':1}));
      svg.append(svgEl('text',{x:48,y:y+4,fill:'#8b9bb5','text-anchor':'end','font-size':12},(10 ** exp).toLocaleString('en-US')));
    }
    task.values.forEach((value,i) => {
      const x = 214 + i * 252; const y = mapY(value);
      const rect = svgEl('rect',{x:x-50,y:animate&&!reduce?286:y,width:100,height:animate&&!reduce?0:286-y,fill:colors[i],rx:4,class:'bar-shape'});
      svg.append(rect);
      const lo = mapY(Math.max(value-task.std[i],value*.08)); const hi = mapY(value+task.std[i]);
      svg.append(svgEl('line',{x1:x,x2:x,y1:hi,y2:lo,stroke:colors[i],'stroke-width':2}));
      for (const ey of [hi,lo]) svg.append(svgEl('line',{x1:x-10,x2:x+10,y1:ey,y2:ey,stroke:colors[i],'stroke-width':2}));
      svg.append(svgEl('text',{x,y:hi-12,fill:i===0?colors[0]:'#e7e9f0','text-anchor':'middle','font-size':23,'font-weight':500},task.labels[i]));
      svg.append(svgEl('text',{x,y:320,fill:'#b7c5de','text-anchor':'middle','font-size':14},names[i]));
      if (animate && !reduce) requestAnimationFrame(() => requestAnimationFrame(() => { rect.setAttribute('y',y); rect.setAttribute('height',286-y); }));
    });
    container.replaceChildren(svg);
    document.getElementById('ablation-summary').textContent = `${task.labels[0]} ms`;
    document.getElementById('ablation-comparison').textContent = `RapidDOC backward time on ${task.name}`;
    document.getElementById('refactor-time').textContent = `${task.labels[1]} ms`;
    document.getElementById('no-reuse-time').textContent = `${task.labels[2]} ms`;
  }
  drawChart(0,false);
  document.querySelectorAll('[data-task]').forEach(button => button.addEventListener('click', () => {
    selectTab(button,button.closest('[role="tablist"]'));
    document.getElementById('ablation-panel').setAttribute('aria-labelledby',button.id);
    drawChart(Number(button.dataset.task));
  }));

  const learning = [
    {src:'assets/fig3_open_loop_imitation_learning.png',caption:'Open-loop objective learning: aligned training behavior and the lowest backward time across all three tasks. Original paper, Fig. 3.',alt:'Open-loop training losses and forward and backward solver time on CartPole, Quadrotor, and Robot Arm'},
    {src:'assets/fig4_closed_loop_imitation_learning.png',caption:'Closed-loop objective learning over state-coupled MPC steps: comparable loss reduction and learned behavior across methods. Original paper, Fig. 4.',alt:'Closed-loop loss curves and teacher and learned trajectories before and after 500 epochs for three robotic tasks'},
    {src:'assets/fig5_figure8_learning.png',caption:'Figure-eight imitation learning: RapidDOC and acados produce comparable training loss decreases; the learned controller follows the reference more closely. Original paper, Fig. 5.',alt:'Figure-eight learning loss compared with acados and reference, before-learning, and after-learning trajectories'}
  ];
  document.querySelectorAll('[data-learning]').forEach(button => button.addEventListener('click', () => {
    selectTab(button,button.closest('[role="tablist"]'));
    const item=learning[Number(button.dataset.learning)];
    const image=document.getElementById('learning-image'); image.src=item.src; image.alt=item.alt; image.removeAttribute('height');
    document.getElementById('learning-full').href=item.src;
    document.getElementById('learning-caption').textContent=item.caption;
    document.getElementById('learning-panel').setAttribute('aria-labelledby',button.id);
  }));

  // Only the visible teaser plays automatically. The full recording remains user-controlled.
  const teaser=document.getElementById('teaser'); const toggle=document.getElementById('teaser-toggle');
  const fullDemo=document.getElementById('full-demo'); let manuallyPaused=false;
  const setPlayState=playing => { toggle.innerHTML=`<span aria-hidden="true">${playing?'Ⅱ':'▶'}</span>`; toggle.setAttribute('aria-label',playing?'Pause demonstration':'Play demonstration'); };
  teaser.addEventListener('play',()=>setPlayState(true)); teaser.addEventListener('pause',()=>setPlayState(false));
  toggle.addEventListener('click',()=>{if(teaser.paused){manuallyPaused=false;teaser.play().catch(()=>setPlayState(false));}else{manuallyPaused=true;teaser.pause();}});
  const videoObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(entry.isIntersecting&&!manuallyPaused&&!reduce)teaser.play().catch(()=>setPlayState(false));else teaser.pause();
  }),{threshold:.25}); videoObserver.observe(teaser);
  fullDemo.addEventListener('play',()=>teaser.pause());
  document.addEventListener('visibilitychange',()=>{if(document.hidden)teaser.pause();});

  document.getElementById('copy-citation').addEventListener('click',async()=>{
    const button=document.getElementById('copy-citation');const text=document.getElementById('citation-text').textContent;
    try{
      await navigator.clipboard.writeText(text);button.textContent='Copied ✓';document.getElementById('copy-status').textContent='Citation copied to clipboard';
      window.setTimeout(()=>{button.innerHTML='Copy citation <span aria-hidden="true">⧉</span>';},2200);
    }catch{
      const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(document.getElementById('citation-text'));selection.removeAllRanges();selection.addRange(range);
      document.getElementById('copy-status').textContent='Citation selected. Use your copy shortcut to copy it.';button.textContent='Text selected';
    }
  });

  const progress=document.querySelector('.page-progress');let queued=false;
  window.addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(()=>{
    const max=document.documentElement.scrollHeight-innerHeight;progress.style.transform=`scaleX(${max>0?scrollY/max:0})`;queued=false;
  });}},{passive:true});
  const sectionObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(entry.isIntersecting){document.querySelectorAll('.nav-links a').forEach(link=>link.classList.toggle('active',link.hash===`#${entry.target.id}`));}
  }),{rootMargin:'-20% 0px -55% 0px'});
  ['method','results','hardware'].forEach(id=>sectionObserver.observe(document.getElementById(id)));
})();
