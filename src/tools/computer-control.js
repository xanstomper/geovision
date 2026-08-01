import robot from 'robotjs';

let controllerReady = false;

try {
  robot.setMouseDelay(50);
  robot.setKeyboardDelay(50);
  controllerReady = true;
} catch (e) {
  console.error('[Control] RobotJS init failed:', e.message);
}

export async function moveMouse(x, y) {
  if (!controllerReady) return { error: 'Controller not available' };
  robot.moveMouse(x, y);
  return { success: true, x, y };
}

export async function click(x, y, button = 'left') {
  if (!controllerReady) return { error: 'Controller not available' };
  if (x && y) robot.moveMouse(x, y);
  robot.mouseClick();
  return { success: true, action: 'click', button };
}

export async function typeText(text) {
  if (!controllerReady) return { error: 'Controller not available' };
  robot.typeString(text);
  return { success: true, text, length: text.length };
}

export async function pressKeys(keys) {
  if (!controllerReady) return { error: 'Controller not available' };
  if (typeof keys === 'string') keys = [keys];
  for (const key of keys) {
    robot.keyToggle(key, 'down');
    await new Promise(r => setTimeout(r, 50));
    robot.keyToggle(key, 'up');
  }
  return { success: true, keys };
}

export async function getScreenInfo() {
  const screen = robot.getScreenSize();
  const mouse = robot.getMousePos();
  return {
    screen: { width: screen.width, height: screen.height },
    mouse: { x: mouse.x, y: mouse.y },
  };
}


export async function takeScreenshotNative(outputPath = './geovision_screenshot_native.png') {
  try {
    const img = robot.screen.capture();
    const { PNG } = await import('node-png').catch(() => ({}));
    return { success: true, path: outputPath };
  } catch (e) {
    return { success: false, fallback: 'Use browser screenshot instead', error: e.message };
  }
}
