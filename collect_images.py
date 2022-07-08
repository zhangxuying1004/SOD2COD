"""
从flickr网站中下载没有版权纠纷的高清图像，
参考https://github.com/prairie-guy/ai_utilities/blob/5dda4f149fa06ee08eda189e4df13da665c05167/image_download.py
"""

import os
import argparse
import shutil
import hashlib
import mimetypes
from glob import glob
from pathlib import Path
from icrawler.builtin import FlickrImageCrawler


def hashfile(path:Path)->str:
    """Create hash of file"""
    blocksize = 65536
    with open(path, 'rb') as f:
        hasher = hashlib.sha512()
        buf = f.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(blocksize)
    return hasher.hexdigest()


def dedupe_images(image_dir:Path)->int:
    """Delete duplicate images from image_dir, also works recursively if there are
    subfolders containing images, OBS: only works in image_files with image extensions"""
    
    image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
    
    images = {}; dups = []
    image_files = [y for x in os.walk(image_dir) for ext in image_extensions for y in glob(os.path.join(x[0], f'*{ext}'))]
    for f in image_files:
        h = hashfile(f)
        if h in images:
            images[h] = images[h] + 1
            dups.append(f)
        else:
            images[h] = 1
    n = len(dups)
    for f in dups:
        Path(f).unlink()
    return n


def image_download(search_text, n_images, label=None, image_dir='dataset', apikey=None):
    """
    Download images from flickr
    """
   
    if label is None: label = search_text
    path = Path.cwd() / image_dir / label
    if Path.exists(path):
        response = input(f"'{label}' exists. Overwrite? [Y/n]: ")
        if response == 'Y': 
            shutil.rmtree(path)
        else: 
            print(f"'{label}' unchanged", end='\r'); 
            exit()

    if apikey == None: 
        print("Flickr requires an apikey: 'https://www.flickr.com/services/api/misc.api_keys.html'"); 
        exit()
    
    crawler = FlickrImageCrawler(apikey, feeder_threads=2, parser_threads=2, downloader_threads=8, storage={'root_dir': path})
    crawler.crawl(tags=search_text, max_num=n_images, tag_mode='all')

    dups = dedupe_images(path)   # Remove duplicates
    print()
    print("**********************************************************")
    print(f"Path:       {path}")
    print(f"Removed:    {dups} duplicate images")
    print(f"Downloaded: {len(list(path.iterdir()))} images")
    print("**********************************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download iamge from flickr') 
    parser.add_argument('--apikey', type=str, default='fd66710d164834d9838fc08cc98e22ac')
    parser.add_argument('--n_images', type=int, default=200)

    """
    'Aquatic-Amimals': 20, 'Flying-Animals': 17, 'Amphibians': 2, 'Terrestrial-Animals': 29, 'Others': 1
    """
    parser.add_argument('--cod10k_category', type=list, 
    default=['Bat-fish', 'Clown-fish', 'Crab', 'Crocodile', 'Crocodile-fish', 'Fish', 'Flounder', 'Frog-fish', 'Ghost-pipefish', 'Leafy-sea-dragon', 'Octopus', 
    'Pagurian', 'Pipefish', 'Scorpion-fish', 'Sea-horse', 'Shrimp', 'Slug', 'Star-fish', 'Stingaree', 'Turtle', 
    'Bat', 'Bee', 'Beetle', 'Bird', 'Bittern', 'Butterfly', 'Cicada', 'Dragonfly', 'Frogmouth', 'Grasshopper', 'Heron', 'Katydid', 'Mantis', 'Mockingbird', 
    'Moth', 'Owl', 'Owlfly', 
    'Frog', 'Toad', 
    'Ant', 'Bug', 'Cat', 'Caterpillar', 'Centipede', 'Chameleon', 'Cheetah', 'Deer', 'Dog', 'Duck', 'Gecko', 'Giraffe', 'Grouse', 'Human', 'Kangaroo', 'Leopard', 
    'Lion', 'Lizard', 'Monkey', 'Rabbit', 'Reccoon', 'Sciuridae', 'Sheep', 'Snake', 'Spider', 'Stick-insect', 'Tiger', 'Wolf', 'Worm', 
    'Others',  
    ])
    
    args = parser.parse_args()

    for i in range(len(args.cod10k_category)):
        search_text = args.cod10k_category[i]
        if search_text == 'Others':
            continue
        # print(search_text)
        image_download(search_text, args.n_images, apikey=args.apikey)
   
  
