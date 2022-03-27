import pytest
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

from core.util.sh import start_web, register_postactions


@pytest.fixture(scope="session", autouse=True)
def setup_django():
    register_postactions()
    start_web()
    sleep(5)


@pytest.fixture(scope="session")
def web_driver(setup_django):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
    yield driver
    driver.quit()
