from test.util import HOMEPAGE


def test_frontend_load(web_driver):
    web_driver.get(HOMEPAGE)
    assert web_driver.title == "Abridge"

    assert "React App" in web_driver.find_element_by_id("root").text
