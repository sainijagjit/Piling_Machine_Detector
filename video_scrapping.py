# -*- coding: utf-8 -*-
import os
import youtube_dl
os.chdir(os.getcwd()+'/Youtube API')
import search 
os.chdir('..')


def video_download(videos,search_term,target_path):
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    os.chdir(target_folder)
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        i=1
        for title,ids in videos:
            ydl.download(['https://www.youtube.com/watch?v='+ids])
            print('{}. {} -Completed'.format(i,title))
            i+=1
            
def search_and_download(search_term,target_path='./video',number_videos=25):
    videos=search.custom_search(search_term,number_videos)
    video_download(videos,search_term,target_path)
    

if __name__ == "__main__":
    search_term='piling machine'
    search_and_download(search_term=search_term,number_videos=10)
    
    
    
    
 
        
        
    