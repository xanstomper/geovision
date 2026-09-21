import { chromium } from 'playwright';
import fs from 'fs';

const BASE = 'http://localhost:3000';
const OUT = '/tmp/zynema_shots';
fs.mkdirSync(OUT, { recursive: true });
const results = [];
const check = (name, ok, extra = '') => { results.push(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' — ' + extra : ''}`); };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });

await page.goto(BASE, { waitUntil: 'networkidle', timeout: 45000 }).catch(() => page.waitForTimeout(3000));
await page.waitForTimeout(2500);

// 1. DOM hierarchy: pages must be siblings inside main, not nested
const structure = await page.evaluate(() => {
  const main = document.getElementById('mainScroll');
  const pages = main ? [...main.querySelectorAll(':scope > .page')] : [];
  return {
    pageIds: pages.map(p => p.id),
    pageListExists: !!document.getElementById('page-list'),
    homeRowsInsideHome: !!document.querySelector('#page-home > #homeRows'),
    moviesInsideHome: !!document.querySelector('#page-home #page-movies'),
    doubleLogoText: document.querySelectorAll('.island-logo').length,
    rowSections: document.querySelectorAll('.row-section').length,
    splashVisible: getComputedStyle(document.getElementById('splashSection')).display !== 'none'
  };
});
check('exactly one #page-home block (no nested pages)', structure.pageIds.filter(id => id.id === 'page-home').length === 1 && !structure.moviesInsideHome);
check('#homeRows lives directly inside #page-home', structure.homeRowsInsideHome);
check('#page-list exists', structure.pageListExists);
check('home rows rendered', structure.rowSections > 0, structure.rowSections + ' sections');
check('splash visible on load', structure.splashVisible);
await page.screenshot({ path: `${OUT}/1-home-desktop.png` });

// 2. Movies / Series / Anime pages with heroes
for (const p of ['movies', 'series', 'anime']) {
  await page.click(`.island-link[data-page="${p}"]`);
  await page.waitForTimeout(1800);
  const heroOk = await page.evaluate(k => {
    const pg = document.getElementById('page-' + k);
    return pg.classList.contains('active') &&
      !!pg.querySelector('.section-hero .sh-title') &&
      pg.querySelectorAll('.grid-card').length > 5;
  }, p);
  check(`${p} page renders hero + grid`, heroOk);
  if (p === 'movies') await page.screenshot({ path: `${OUT}/2-movies-desktop.png` });
}

// 3. Open a title, add to My List, verify persistence + page
await page.click('.island-link[data-page="home"]');
await page.waitForTimeout(1500);
await page.click('.row-card');
await page.waitForTimeout(1500);
const panelOpen = await page.evaluate(() => document.getElementById('cinematicOverlay').classList.contains('open'));
check('details panel opens', panelOpen);
await page.click('#coWatchlistBtn');
await page.waitForTimeout(400);
const saved = await page.evaluate(() => ({
  ls: JSON.parse(localStorage.getItem('zynema:mylist') || '[]').map(x => x.title),
  btn: document.getElementById('coWatchlistBtn').classList.contains('watchlist-saved')
}));
check('watchlist button saves title', saved.ls.length === 1 && saved.btn, JSON.stringify(saved.ls));
await page.click('#coClose');
await page.waitForTimeout(400);
await page.click('.island-link[data-page="list"]');
await page.waitForTimeout(700);
const listOk = await page.evaluate(() => ({
  active: document.getElementById('page-list').classList.contains('active'),
  cards: document.querySelectorAll('#listGrid .grid-card').length,
  count: document.getElementById('listCount').textContent
}));
check('My List shows saved title', listOk.active && listOk.cards === 1, listOk.count);
await page.screenshot({ path: `${OUT}/3-mylist-desktop.png` });

// 4. Remove from list → empty state
await page.click('#listGrid .grid-card');
await page.waitForTimeout(1300);
await page.click('#coWatchlistBtn');
await page.waitForTimeout(400);
await page.click('#coClose');
await page.click('.island-link[data-page="list"]');
await page.waitForTimeout(600);
const emptyOk = await page.evaluate(() => {
  const e = document.getElementById('listEmpty');
  return document.querySelectorAll('#listGrid .grid-card').length === 0 && getComputedStyle(e).display === 'flex';
});
check('removal works + empty state shows', emptyOk);

// 5. Search overlay + Escape closes it (hover top edge first to reveal nav, as a user would)
await page.mouse.move(960, 10);
await page.waitForTimeout(700);
await page.click('#searchWrapper');
await page.waitForTimeout(500);
const searchOpen = await page.evaluate(() => document.getElementById('searchOverlay').classList.contains('open'));
await page.keyboard.press('Escape');
await page.waitForTimeout(400);
const searchClosed = await page.evaluate(() => !document.getElementById('searchOverlay').classList.contains('open'));
check('search overlay opens + Escape closes', searchOpen && searchClosed);

// 6. Mobile viewport
const mob = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
await mob.goto(BASE, { waitUntil: 'networkidle', timeout: 45000 }).catch(() => mob.waitForTimeout(3000));
await mob.waitForTimeout(2500);
const mobOk = await mob.evaluate(() => {
  const island = document.getElementById('dynamicIsland');
  const r = island.getBoundingClientRect();
  return r.width <= window.innerWidth + 2 && document.querySelectorAll('.row-section').length > 0;
});
check('mobile layout renders, nav fits viewport', mobOk);
await mob.screenshot({ path: `${OUT}/4-home-mobile.png` });
await mob.click('.island-link[data-page="movies"]').catch(() => {});
await mob.waitForTimeout(1500);
await mob.screenshot({ path: `${OUT}/5-movies-mobile.png` });

check('no page JS errors during run', errors.length === 0, errors.slice(0, 3).join(' | '));
await browser.close();
console.log(results.join('\n'));
fs.writeFileSync(`${OUT}/results.txt`, results.join('\n'));
