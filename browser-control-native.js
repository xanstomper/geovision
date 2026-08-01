#!/usr/bin/env node
"""
GeoVision Native Browser Control
AppleScript + CLI for macOS browser and screen control
"""

import { exec } from 'child_process';
import { promisify } from 'util';
const execAsync = promisify(exec);

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

class NativeBrowserControl {
  async getActiveTab() {
    try {
      const { stdout } = await execAsync(
        "osascript -e 'tell application \"Google Chrome\" to get URL of active tab of front window'"
      );
      const { stdout: title } = await execAsync(
        "osascript -e 'tell application \"Google Chrome\" to get title of active tab of front window'"
      );
      return { url: stdout.trim(), title: title.trim() };
    } catch (e) {
      return { error: e.message };
    }
  }

  async openUrl(url) {
    await execAsync(`open "${url}"`);
    await delay(3000);
    return { success: true, url };
  }

  async navigateMap(lat, lng, zoom) {
    const url = `https://www.google.com/maps/@${lat},${lng},${zoom || 15}z`;
    await this.openUrl(url);
    return { success: true, url };
  }

  async screenshot(outputPath) {
    try {
      await execAsync(`screencapture -x ${outputPath}`);
      return { success: true, path: outputPath };
    } catch (e) {
      return { success: false, error: e.message };
    }
  }

  async moveMouse(x, y) {
    await execAsync(`cliclick c=${x},${y}`).catch(() => {
      execAsync(`osascript -e 'tell application \"System Events\" to do shell script \"cliclick c=${x},${y}\"'`);
    });
    return { success: true, x, y };
  }

  async click(x, y, button = 'left') {
    await execAsync(`osascript -e 'tell application \"System Events\" to click at {${x}, ${y}}'`);
    return { success: true, x, y, button };
  }

  async typeText(text) {
    const safe = text.replace(/"/g, '\\"');
    await execAsync(`osascript -e 'tell application \"System Events\" to keystroke \"${safe}"'`);
    return { success: true, text };
  }
}

export default new NativeBrowserControl();
