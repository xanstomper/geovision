"""
Browser Automation Module
Controls browser for live verification and interaction
"""

import time
import logging
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BrowserAutomation:
    """
    Cross-browser automation for geolocation verification
    Supports multiple backends: Selenium, Playwright, MCP browser
    """
    
    BACKENDS = {
        "selenium": "Selenium WebDriver",
        "playwright": "Playwright",
        "mcp": "MCP Browser Control",
        "pyautogui": "System-level automation",
        "requests": "Simple Requests Backend"
    }
    
    def __init__(self, backend: str = "requests"):
        self.backend = backend
        self.driver = None
        self.active = False
    
    def connect(self) -> bool:
        try:
            if self.backend == "mcp":
                return self._connect_mcp()
            elif self.backend == "selenium":
                return self._connect_selenium()
            elif self.backend == "playwright":
                return self._connect_playwright()
            elif self.backend == "pyautogui":
                return self._connect_pyautogui()
            elif self.backend == "requests":
                return self._connect_requests()
        except Exception as e:
            logger.error(f"Browser connection error: {e}")
        return False
    
    def _connect_requests(self) -> bool:
        self.active = True
        return True
    
    def _connect_mcp(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(["mcp", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.active = True
                logger.info("MCP browser backend available")
                return True
        except FileNotFoundError:
            logger.warning("MCP not found, falling back to pyautogui")
        return False
    
    def _connect_selenium(self) -> bool:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(options=options)
            self.active = True
            return True
        except Exception as e:
            logger.error(f"Selenium init error: {e}")
            return False
    
    def _connect_playwright(self) -> bool:
        try:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.driver = self.playwright.chromium.launch(headless=True)
            self.active = True
            return True
        except Exception as e:
            logger.error(f"Playwright init error: {e}")
            return False
    
    def _connect_pyautogui(self) -> bool:
        import pyautogui
        pyautogui.FAILSAFE = True
        self.active = True
        logger.info("PyAutoGUI backend active")
        return True
    
    def navigate(self, url: str, wait: float = 2.0):
        if self.backend == "selenium" and self.driver:
            self.driver.get(url)
            time.sleep(wait)
        elif self.backend == "playwright" and self.driver:
            page = self.driver.new_page()
            page.goto(url)
            time.sleep(wait)
            return page
        elif self.backend == "pyautogui":
            import webbrowser
            webbrowser.open(url)
            time.sleep(wait)
        elif self.backend == "requests":
            logger.info(f"Verification URL generated: {url}")
            return url
    
    def screenshot(self, output_path: Optional[str] = None) -> Optional[bytes]:
        try:
            if self.backend == "selenium" and self.driver:
                png = self.driver.get_screenshot_as_png()
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(png)
                return png
            elif self.backend == "playwright" and self.driver:
                page = self.driver.new_page()
                png = page.screenshot(full_page=True)
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(png)
                return png
            elif self.backend == "pyautogui":
                import pyautogui
                screenshot = pyautogui.screenshot()
                if output_path:
                    screenshot.save(output_path)
                return screenshot.tobytes()
            elif self.backend == "requests":
                logger.info("Requests backend cannot take screenshots.")
                return None
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
        return None
    
    def click(self, x: int, y: int):
        if self.backend == "selenium" and self.driver:
            from selenium.webdriver.common.action_chains import ActionChains
            ActionChains(self.driver).move_to_element_with_offset(
                self.driver.find_element("tag name", "body"), x, y
            ).click().perform()
        elif self.backend == "pyautogui":
            import pyautogui
            pyautogui.click(x, y)
    
    def type_text(self, text: str):
        if self.backend == "selenium" and self.driver:
            from selenium.webdriver.common.keys import Keys
            active = self.driver.switch_to.active_element
            active.send_keys(text)
        elif self.backend == "pyautogui":
            import pyautogui
            pyautogui.typewrite(text)
    
    def close(self):
        try:
            if self.backend == "selenium" and self.driver:
                self.driver.quit()
            elif self.backend == "playwright":
                self.playwright.stop()
            self.active = False
        except:
            pass