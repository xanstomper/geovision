import { chromium } from 'playwright';
import fs from 'node:fs';
const output='/tmp/zynema-final-verification';
fs.mkdirSync(output,{recursive:true});
const lines=[];
function log(message){lines.push(message);console.log(message);fs.writeFileSync(`${output}/results.txt`,lines.join('\n')+'\n');}
function check(name,condition,details){log(`${condition?'PASS':'FAIL'} ${name}${details?' '+JSON.stringify(details):''}`);if(!condition)throw Error(name);}
const browser=await chromium.launch();
let page;
try {
  page=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  page.setDefaultTimeout(10000);
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://localhost:3000',{waitUntil:'domcontentloaded',timeout:15000});
  await page.locator('#homeRows .row-card').first().waitFor();
  check('fresh-load home renders',await page.locator('#homeRows .row-card').count()>0);
  for(const width of [320,390,760,768,1440,1920]){
    await page.setViewportSize({width,height:844});
    const bounds=await page.evaluate(()=>[...document.querySelectorAll('.island-logo,.island-link,#searchWrapper,#profileBtn')].map(n=>{
      const r=n.getBoundingClientRect();return {name:n.id||n.dataset.page,left:r.left,right:r.right,width:r.width};
    }));
    check(`navigation fits ${width}px`,bounds.every(r=>r.left>=0&&r.right<=width&&r.width>0),bounds);
  }
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:`${output}/mobile-home.png`,timeout:10000});
  // Use the second poster, away from the carousel arrows and central Play control.
  await page.locator('.row-card').nth(1).click({position:{x:70,y:40}});
  await page.locator('#cinematicOverlay.open').waitFor();
  const hit=await page.locator('#coClose').evaluate(n=>{
    const r=n.getBoundingClientRect(),top=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
    return {reachable:n===top||n.contains(top),interceptor:top?.outerHTML.slice(0,200)};
  });
  check('mobile modal close button is not covered',hit.reachable,hit);
  await page.locator('#coWatchlistBtn').click();
  await page.locator('#coClose').click();
  await page.locator('#cinematicOverlay').waitFor({state:'hidden'});
  await page.locator('.island-link[data-page="list"]').click();
  check('saved title renders in My List',await page.locator('#listGrid .grid-card').count()===1);
  check('singular list count',await page.locator('#listCount').innerText()==='1 title');
  const clearance=await page.evaluate(()=>({titleTop:document.querySelector('#page-list h2').getBoundingClientRect().top,navBottom:document.getElementById('dynamicIsland').getBoundingClientRect().bottom}));
  check('My List title clears mobile navigation',clearance.titleTop>=clearance.navBottom,clearance);
  await page.screenshot({path:`${output}/mobile-list.png`,timeout:10000});
  await page.locator('#searchWrapper').click();
  await page.locator('#searchOverlay').waitFor({state:'visible'});
  check('search opens after modal dismissal',await page.locator('#searchOverlayInput').evaluate(n=>document.activeElement===n));
  await page.keyboard.press('Escape');
  await page.locator('#searchOverlay').waitFor({state:'hidden'});
  check('Escape closes search',true);
  await page.locator('#listGrid .grid-card').click({position:{x:70,y:40}});
  await page.locator('#coWatchlistBtn').click();
  await page.locator('#coClose').click();
  await page.locator('#listEmpty').waitFor({state:'visible'});
  check('removal restores empty state',await page.locator('#listGrid .grid-card').count()===0);
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#searchWrapper').click();
  await page.locator('#searchOverlay').waitFor({state:'visible'});
  await page.keyboard.press('Escape');
  await page.locator('#searchOverlay').waitFor({state:'hidden'});
  check('desktop search opens and closes',true);
  await page.screenshot({path:`${output}/desktop-list.png`,timeout:10000});
  check('no uncaught page JavaScript errors',errors.length===0,errors);
  log('ALL CHECKS PASSED');
}catch(error){log(`FAILED: ${error.stack}`);process.exitCode=1;}
finally{await browser.close();}
