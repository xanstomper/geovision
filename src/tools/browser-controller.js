import { chromium } from 'playwright';
import fs from 'fs/promises';
import { exec } from 'child_process';
import { promisify } from 'util';
const execAsync = promisify(exec);

let browserInstance = null;
let page = null;

async function getBrowser() {
  if (!browserInstance) {
    browserInstance = await chromium.launch({ headless: false, args: ['--start-maximized'] });
  }
  if (!page) {
    const contexts = browserInstance.contexts();
    if (contexts.length > 0) {
      const pages = contexts[0].pages();
      if (pages.length > 0) page = pages[0];
      else page = await contexts[0].newPage();
    } else {
      page = await browserInstance.newPage();
    }
  }
  return { browser: browserInstance, page };
}

export async function openGoogleMaps({ query, lat, lng, zoom }) {
  try {
    const { page: p } = await getBrowser();
    const zoomVal = zoom || 15;
    let url = 'https://www.google.com/maps/';
    if (query) {
      url = `https://www.google.com/maps/search/${encodeURIComponent(query)}`;
    } else if (lat && lng) {
      url = `https://www.google.com/maps/@${lat},${lng},${zoomVal}z`;
    }
    await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await p.waitForTimeout(3000);
    return { success: true, url: p.url(), message: `Opened: ${query || `${lat},${lng}`}` };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

export async function getCurrentTab() {
  try {
    const { page: p } = await getBrowser();
    return { url: p.url(), title: await p.title() };
  } catch (e) {
    return { error: 'No active browser session', details: e.message };
  }
}

export async function navigateMap(lat, lng, zoom) {
  try {
    const { page: p } = await getBrowser();
    const zoomVal = zoom || 15;
    await p.goto(`https://www.google.com/maps/@${lat},${lng},${zoomVal}z`, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await p.waitForTimeout(2000);
    return { success: true, coords: `${lat},${lng}` };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

export async function takeScreenshot({ fullScreen = false, outputPath = './geovision_screenshot.png' }) {
  try {
    const out = outputPath.endsWith('.png') ? outputPath : outputPath + '.png';
    if (fullScreen) {
      await execAsync(`gnome-screenshot -f ${out}`);
    } else {
      const { page: p } = await getBrowser();
      await p.screenshot({ path: out, fullPage: false });
    }
    const stats = await fs.stat(out);
    return { success: true, path: out, size: stats.size };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

async function closeBrowser() {
  if (browserInstance) {
    await browserInstance.close();
    browserInstance = null;
    page = null;
  }
}
