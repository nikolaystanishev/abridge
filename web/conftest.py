from time import sleep
import pytest
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from core.util.sh import start_web, register_postactions


@pytest.fixture(scope="session", autouse=True)
def setup_django():
    register_postactions()
    start_web()


@pytest.fixture(scope="session")
def web_driver(setup_django):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    yield driver
    driver.quit()
