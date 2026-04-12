import asyncio
from playwright.async_api import async_playwright
from file_manager.file_manager import save_file


def get_dom_from_website(website_link, path_file):
    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=50)
            page = await browser.new_page()
            await page.goto(website_link)
            content = await page.content()
            save_file(content, path_file)
            print("Test completed sucessfully")
            await browser.close()

    asyncio.run(main())