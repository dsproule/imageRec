from selenium import webdriver
from bs4 import BeautifulSoup
import random 
import time
import os
import requests
import math
from threading import Thread

class Scraper():
	def __init__(self, folder):
		self.folder = folder
		self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15"}
		self.unsplash_c, self.dreamtime_c = 400, 200
		self.pixabay_c, self.shutter_c = 400, 400
		self.shutter_r = .4
		self.u_chunks = 30
		self.branches = [self._unsplash, self._dreamtime, self._sel_dep_helper]
		self.sel_inst = [self._pixabay, self._shutterstock]
		self.image_urls = []

		if self.folder not in os.listdir():
			os.mkdir(self.folder)

	# Retrieves all the image urls and returns them for further processing
	def run(self):
		threads = []
		self.driver = webdriver.Safari()
		print("Starting the scraping threads:\n")
		for branch in self.branches:
			threads.append(Thread(target=branch))

		[thread.start() for thread in threads]
		[thread.join() for thread in threads]

		print(f'{len(self.image_urls)} images retreieved')
		
		self.driver.close()
		return self.image_urls

	# Some websites would block the requests library so a headless browser is created instead to gather all the images
	def _sel_dep_helper(self):
		for inst in self.sel_inst:
			inst()


	# WARNING: All web scraping branches are unique to the websites' configurations at the time of writing the script and may not work without maintenence because websites may update their structure
	def _unsplash(self):
		# unsplash branch
		print("  Scraping unsplash...")
		unsplash_ub = (self.unsplash_c // self.u_chunks) + 2
		self.c_total = len(os.listdir(self.folder))
		r = requests.Session()
		r.headers.update(self.headers)
		for i in range(0, unsplash_ub):
			fetch = r.get(f'https://unsplash.com/napi/search/photos?query={self.folder}&per_page={self.u_chunks}&page={i}&xp=').json()['results']
			
			for result in fetch:
				self.image_urls.append(result['urls']['raw'])
		print(' Unsplash done.')

	def _dreamtime(self):
		print("  Scraping dreamtime...")
		# downloads 80 at a time
		dreamtime_ub = (self.dreamtime_c // 80) + 1

		if dreamtime_ub <= 3:
			dreamtime_ub = 4
		# Where the cookie assignment would go
		r = requests.Session()
		r.headers.update(self.headers)
		for i in range(2, dreamtime_ub+1):
			time.sleep(random.random()*2 + .5)
			fetch = r.get(f'https://www.dreamstime.com/photos-images/{self.folder}.html?pg={i}', headers=self.headers)

			soup = BeautifulSoup(fetch.text, 'lxml')
			imgs = soup.find('main').find('div', class_='container-fluid thb-large-box').find_all('img')
			for img_brick in imgs:
				self.image_urls.append(img_brick['data-src'])
		print(' Dreamtime done.')

	def _pixabay(self):
		print("  Scraping pixabay...")		
		pixabay_ub = (self.pixabay_c // 100) + 2
		if pixabay_ub <= 2:
			pixabay_ub = 3
		
		for i in range(1, pixabay_ub):
			self.driver.get(f'https://pixabay.com/images/search/{self.folder}/?pagi={i}&')
			html = self.driver.page_source

			soup = BeautifulSoup(html, 'lxml')
			results = soup.find('div', class_='container--HcTw2').find_all('img')
			for result in results:
				if result.has_attr('data-lazy-src'):
					pr = result['data-lazy-src']
				else:
					pr = result['src']
				self.image_urls.append(pr)		
		print(' Pixabay done.')


	def _get_shutter_ids(self, page_content):
		ids_ = []
		soup = BeautifulSoup(page_content, 'lxml')
		im_group = soup.find('div', class_='jss150 jss152 jss154')
		for img in im_group.find_all('a'):
			full_im_id = img['href']
			im_id, pre = full_im_id[full_im_id.rfind('-'):], full_im_id[:full_im_id.rfind('-')]
			ids_.append('https://image.shutterstock.com'+pre+'-260nw'+im_id+'.jpg')
		return ids_

	def _shutterstock(self):
		print("  Scraping shutterstock...")
		shutter_ub = (self.shutter_c // 106) + 2 
		shutter_ub = 3 if shutter_ub < 3 else shutter_ub
		for i in range(1, shutter_ub):
			self.driver.get(f'https://www.shutterstock.com/search/{self.folder}?page={i}')
			src = self.driver.page_source

			[self.image_urls.append(im_url) for im_url in self._get_shutter_ids(src)]
		print(' Shutterstock done.')