import os
import requests
import time

# 进度条模块
def progressbar(url):
    start = time.time() #下载开始时间
    response = requests.get(url, stream=True)
    size = 0    #初始化已下载大小
    chunk_size = 1024  # 每次下载的数据大小
    content_size = int(response.headers['content-length'])  # 下载文件总大小
    try:
        if response.status_code == 200:   #判断是否响应成功
            print('Start download,[File size]:{size:.2f} MB'.format(size = content_size / chunk_size /1024))   #开始下载，显示下载文件大小
            filepath = 'last.pt'  #设置图片name，注：必须加上扩展名
            with open(filepath,'wb') as file:   #显示进度条
                for data in response.iter_content(chunk_size = chunk_size):
                    file.write(data)
                    size +=len(data)
                    print('\r'+'[下载进度]:%s%.2f%%' % ('>'*int(size*50/ content_size), float(size / content_size * 100)) ,end=' ')
        end = time.time()   #下载结束时间
        print('Download completed!,times: %.2f秒' % (end - start))  #输出下载用时时间
    except:
        print('Error!')

def main():
    #模型文件
    url = 'http://www.yuajiahao.info/wp-content/MyFile/last.pt'
    progressbar(url)

if __name__ == '__main__':
    main()