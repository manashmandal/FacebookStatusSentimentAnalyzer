from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from requests import get
from lxml import html

class FacebookSpider():
    def __init__(self, userid, password):
        self.userid = userid
        self.password = password
        self.uname_xpath = "//input[@id='email']"
        self.pass_xpath = "//input[@id='pass']"
        self.loginbutton_id = "loginbutton"


        #Init values
        self.facebook = 'https://www.facebook.com'
        self.delay_btn_scroll = 0
        self.scroll_count = 100
        self.browser = None
        self.profile = ''
        self.scraped_urls = []
        self.scraped_posts = []
        self.profile_posts = ''
        self.profile_videos = ''
        self.profile_photos = ''
        self.profile_source = ''
        self.profile_id = ''


    def set_scroll_count(self, count):
        self.scroll_count = count

    def set_profile_to_scrape(self, profile_url):
        self.profile_id = profile_url
        self.profile_videos = profile_url + '/videos'
        self.profile_photos = profile_url + '/photos'
        self.profile_posts = profile_url + '/posts'
        self.profile = self.facebook + '/' + profile_url

    def set_webdriver_path(self, path):
        self.webdriver_path = path


    def set_delay_between_scroll(self, delay):
        self.delay_btn_scroll = delay


    def extract_post_urls(self):
        # Getting the chrome browser and the elements
        self.browser = webdriver.Chrome(self.webdriver_path)

        self.browser.get(self.facebook)

        uname_element = self.browser.find_element_by_xpath(self.uname_xpath)
        pass_element = self.browser.find_element_by_xpath(self.pass_xpath)
        loginbutton_element = self.browser.find_element_by_id(self.loginbutton_id)

        # Logging in
        uname_element.send_keys(self.userid)
        pass_element.send_keys(self.password)
        loginbutton_element.click()

        # self.browser.implicitly_wait(1)

        body = self.browser.find_element_by_tag_name('body')
        print("Got the body tag")

        for i in range(1):
            print("Clicking the body")
            body.send_keys(Keys.ESCAPE)
            # body.click()

        # Visiting the url
        self.browser.get(self.profile)
        print("Visiting the profile")

        # Getting the body for scrolling
        body = self.browser.find_element_by_tag_name('body')
        print("Got the body tag again")

        # Quitting the annoying desktop notification popups
        for i in range(10):
            body.send_keys(Keys.ESCAPE)
            # body.click()

        for i in range(self.scroll_count):
            body.send_keys(Keys.PAGE_DOWN)
            print("Scroll Count: {0}".format(i + 1))
            self.browser.implicitly_wait(self.delay_btn_scroll)

        # Waaiting
        # self.browser.implicitly_wait(5)

        # Finally getting the source
        self.profile_source = self.browser.page_source

        # Makin' a soup
        profile_soup = BeautifulSoup(self.profile_source, 'lxml')

        links = ['https://www.facebook.com' + l['href'] for l in profile_soup.find_all('a', attrs={'class': '_5pcq'}) if ('#' not in l['href']) and (self.profile_id in l['href'])]

        self.browser.close()

        return links


    def extract_text(self, urls):
        text_status = []
        for url in urls:
            print("Requesting: {0}".format(url))
            page = get(url).content
            tree = html.fromstring(page)
            container = tree.xpath("//div[contains(@class, '_5pbx userContent')]")[0].text_content()
            text_status.append(container)
            print("Status Collected: {0}".format(url))

        return text_status

