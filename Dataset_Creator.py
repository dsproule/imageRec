import Scrapers as scrape
import numpy as np
import time
import os
import cv2
import asyncio
import aiohttp
import aiofiles
import math
import json

'''
    Creator.comb(folder, inc=4):
        Opens the data sequentially in increments of inc. Size of the images can be changed by modifying the
        Creator.scale value. Default is 150

    Creator.build(folder):
        Scrapes websites and runs first pass of image downloads. Then places them all into a folder and runs
        the clean up function.

    Creator.salt_and_pepper(folder, ub, probability=.3, lb=0):
        --> Built to work from left to right
        Probability cannot exceed 1. Applies salt and pepper noise to images from [lb:ub]

    Creator.rotate(folder, ub, lb=len(os.list), sel='random'):
        --> Built to work from right to left
        
    Creator.empty(folder):
        Deletes all files in a folder

    Creator.view(folder, im_name):
        Displays an image from dataset to user

    Creator.convert_to_npy([folder1, folder2, folder3...]):
        Assigns a one-hot-vector to each image and then saves them in a npy file. Also creates a validation set.
        Vector is assigned in order of name. folder1 is [1, 0], folder2 is [0, 1], etc.

    Creator.reorder(folder):
        Renames files to be 0.ext - len(folder).ext
'''

class Creator(object):
    def __init__(self):
        self.dl_total, self.scale = 0, 150
        self.to_d = set()
        self.temp_validation_set, self.validation_set, self.training_set = [], [], []
        self.labels, self.count = {}, {}
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15"}

        with open("left_off.json", 'r') as f:
            self.lo = json.load(f)

    def comb(self, folder='.', inc=4):         # Reviews data allowing for selection to be removed
        images = [file for file in os.listdir(folder) if (file[-4:] == '.jpg')]
        images.sort()
       
        folder = '' if folder == '.' else folder + '/'
        if folder in self.lo.keys():
            images = images[self.lo[folder]:]
        else:
            self.lo[folder] = 0

        if self.lo[folder] == -1:
            print(f"{folder[:-1]} has already been checked")
            return
        for im_p in range(0, len(images), inc):
            if len(images) - im_p < inc:       # Makes it so the images fit if it's not divisible
                inc = len(images) - im_p

            bundle = tuple([cv2.resize(cv2.imread(f'{folder}{images[i]}')[:-20], (self.scale, self.scale)) for i in range(im_p, im_p+inc)])
            bundle = np.concatenate(bundle, axis=1)         # Bundles current img and next few by inc

            while True:
                cv2.imshow(f"{im_p}-{im_p+inc}/{len(images)}", bundle)

                keypress = cv2.waitKey(0)

                if keypress == 32:      # spacebar
                    break
                if keypress > 48 and keypress < 58:     # 1-9
                    cv2.destroyAllWindows()         # Loads number into to_d list and displays it
                    im_2d = f'{folder}{images[im_p + (keypress - 49)]}'
                    cv2.imshow('To delete', cv2.resize(cv2.imread(im_2d)[:-20], (self.scale, self.scale)))
                    self.to_d.add(im_2d)
                    cv2.waitKey(500)
                if keypress == 99:       # c --> complete deletion
                    self.lo[folder] = im_p + self.lo[folder] - len(self.to_d)
                    self._delete()      # exits and deletes all fils in to_d
                    print(f"There are {len(os.listdir(folder))} images left in {folder}")
                    return
                if keypress == 97:
                    cv2.destroyAllWindows()         # Loads number into to_d list and displays it
                    for i in range(1, 5):
                        self.to_d.add(f'{folder}{images[im_p + i]}')
                    break
                if keypress == 98:      # b --> bail
                    return

            cv2.destroyAllWindows()

        self.lo[folder] = -1
        self._delete() 

    def salt_and_pepper(self, folder, ub, probability=.3, lb=0):        # Adds salt and pepper noise
        im_group = os.listdir(folder)[lb:ub]
        for im in im_group:
            im_path = f'{folder}/{im}'
            im = cv2.resize(cv2.imread(im_path), (self.scale, self.scale))
            for col in range(im.shape[0]):
                for row in range(im.shape[1]):
                    rand = self._random(1)
                    if rand <= probability:
                        if rand > .5:
                            im[col][row] = [255, 255, 255]
                        else:
                            im[col][row] = [0, 0, 0]
            new_name = str(max([int(im[:im.rfind('.')]) for im in os.listdir(folder)]) + 1) + im_path[im_path.rfind('.'):]
            cv2.imwrite(f'{folder}/{new_name}', im)
            print(f"{new_name} added to {folder}")

    def rotate(self, folder, lb, ub=0, sel='random'):
        ub = len(os.listdir(folder)) - ub
        lb *= -1

        for im in os.listdir(folder)[lb:ub]:
            tmp_sel = math.floor(self._random(3)) if sel == 'random' else sel
            im_path = f'{folder}/{im}'
            img = cv2.rotate(cv2.imread(im_path), tmp_sel)

            new_name = str(max([int(im[:im.rfind('.')]) for im in os.listdir(folder)]) + 1) + im_path[im_path.rfind('.'):]
            cv2.imwrite(f'{folder}/{new_name}', img)
            print(f"{new_name} added to {folder}")

    def reorder(self, folder):       # Renames all files to be 0-len(ims)
        for file in os.listdir(folder):
            os.rename(f'{folder}/{file}', f'{folder}/t{file}')

        for i, file in enumerate(os.listdir(folder)):
            ext = file[file.rfind('.'):]
            os.rename(f'{folder}/{file}', f'{folder}/{i}{ext}')

    def build(self, folder):        # Creates the dataset to the desired location
        urls = scrape.Scraper(folder).run()
        self.total, self.dl_total = len(urls), 0
        delay = self._random(3, 8)
        print(f"Pausing program for {round(delay, 2)} seconds...")
        time.sleep(delay)

        print(f"Downloading {self.total} images... to {folder}")
        asyncio.run(self._async_helper_fn(urls, folder))
        
        self._cleanup(folder)
        print(f"Raw dataset has been built to {folder}")
    
    def empty(self, folder):        # Clears a dataset
        if folder == ' ' or folder == '.' or folder == '':
            return
        print(f'{len(os.listdir(folder))} images will be deleted from {folder}')
        for img in os.listdir(folder):
            os.remove(f'{folder}/{img}')
        print(f'ls {folder}: {os.listdir(folder)}')

    def view(self, folder, im_name):         # Selects a random image from the dataset and displays it
        img = f'{im_name}'
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[:-20]
        img = cv2.resize(img, (self.scale, self.scale))
        cv2.imshow(f"{img}", img)
        cv2.waitKey(0)

    def convert_to_npy(self, datasets):         # Convert datasets to npy
        if len(datasets) < 2:
            print(datasets)
            raise Exception("InvalidAction: Less then two datasets loaded in")
        [self.reorder(dataset) for dataset in datasets]
        self._select_validation_set(datasets)
        for dataset in datasets:
            self.labels[dataset] = len(self.labels)
            one_hot = np.eye(len(datasets))[self.labels[dataset]]
            self.count[dataset]  = 0
            for im_label in os.listdir(dataset):
                try:
                    if f'{dataset}/{im_label}' in self.temp_validation_set:
                        continue
                    img = cv2.imread(f'{dataset}/{im_label}', cv2.IMREAD_GRAYSCALE)[:-20] / 255.0
                    img = cv2.resize(img, (self.scale, self.scale))
                    self.training_set.append([np.array(img), one_hot])
                    if self.count[dataset] % 10 == 0:
                        print(f'Img {self.count[dataset]} of {dataset} converted')
                    self.count[dataset] += 1
                except:
                    print(f"IMAGE: {im_label} in {dataset} broke and was not processed")

        np.random.shuffle(self.training_set)
        np.save(f"{'_'.join([key[0:3] for key in self.labels.keys()])}_train_set.npy", self.training_set)
        self._finalize_validation_set()

        for dataset in datasets:
            print(f"{dataset} contained {self.count[dataset]} images")

    async def _async_helper_fn(self, urls, folder):
        async with aiohttp.ClientSession(trust_env=True) as session:
            reqs = []
            ind = 0
            for url in urls:
                req = asyncio.ensure_future(self._download(session, url, ind, folder))
                reqs.append(req)
                ind +=1 

            await asyncio.gather(*reqs)
            print("Done")

    async def _download(self, session, url, ind, folder):
        async with session.get(url) as response:
            ext = url[url.rfind('.'):]
            if ext not in ['.png', '.jpg', 'jpeg']:
                ext = '.jpg'
            f = await aiofiles.open(f'{folder}/{ind}{ext}', mode='wb')
            await f.write(await response.read())
            await f.close()
            print(f"Downloaded img {self.dl_total} for {folder}")
            self.dl_total += 1
            if self.dl_total % 800 == 0:
                delay = self._random(3, 27)     # Pauses from 27-30 seconds
                print(f"Imposing pause for {round(delay, 2)} seconds")
                time.sleep(delay)

    def _delete(self):        # Deletes the images queued up to be removed
        for img in self.to_d:
            os.remove(img)
        print(f"{len(self.to_d)} images deleted")
        self.to_d = set()
        with open("left_off.json", 'w') as f:
            f.write(json.dumps(self.lo))

    def _cleanup(self, folder):         # Gets rid of all images smaller than 1000 bytes
        print(f"Cleaning up {folder}...")
        for ind, file in enumerate(os.listdir(f'{folder}/')):
            path = f'{folder}/{file}'
            if ind % 50 == 0:
                print(f"Image {ind} just passed")
            if os.path.getsize(path) < 1000:
                self.to_d.add(path)
            try: 
                cv2.resize(cv2.imread(path), (1, 1))
            except:
                self.to_d.add(path)
        self._delete()
        [print(f'{deleted_f} was removed') for deleted_f in self.to_d]

        os.system(f'cd {folder}/; mkdir tmp; for f in ./*.png; do pngfix --strip=color --out=tmp/"$f" "$f"; done')  # Cleanup pngs
        for file in os.listdir(f'{folder}/tmp/'):
            try: os.remove(f'{folder}/{file}')
            except FileNotFoundError: print(f'File: {file} does not exist in {folder}')
            os.replace(f'{folder}/tmp/{file}', f'{folder}/{file}')
        os.rmdir(f'{folder}/tmp')

    def _select_validation_set(self, datasets):
        with open("validation_images.txt", 'w') as f:
            for dataset in datasets:
                for i in range(int(.1 * sum([len(os.listdir(dataset)) for dataset in datasets]))):
                    ind = self._random(len(os.listdir(dataset)))
                    self.temp_validation_set.append(f'{dataset}/{os.listdir(dataset)[math.floor(ind)]}')
                    f.write(f'{dataset}/{os.listdir(dataset)[math.floor(ind)]}\n')


    def _finalize_validation_set(self):     # Turns them into npy set
        print("Beginning Finalization of validation set...")
        for im_label in self.temp_validation_set:
            ind = self.labels[im_label[:im_label.rfind('/')]]
            one_hot = np.eye(len(self.labels))[ind]
            
            img = cv2.imread(im_label, cv2.IMREAD_GRAYSCALE) / 255.0
            img = cv2.resize(img, (self.scale, self.scale))
            self.training_set.append([np.array(img), one_hot])

        np.random.shuffle(self.training_set)
        np.save(f"{'_'.join([key[0:3] for key in self.labels.keys()])}_val_set.npy", self.training_set)

    def _random(self, offset, base=0):
        return np.random.random(1)[0] * offset + base
    
if __name__ == '__main__':
    c = Creator()
    datasets = ['apple', 'watermelon', 'banana']
    c.scale = 100
    c.convert_to_npy(datasets)