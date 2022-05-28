import copy
import queue
import threading
from PyQt5.QtCore import QThread, pyqtSignal
import time
import numpy as np
from eit import Eit

TIME_LIMIT = 100
exitFlag = 0
Que = queue.Queue()
queueLock = threading.Lock()
ID, ID_SUM = 0, 0
threads = []
save_data_list = []

class Timer(QThread):
    ''' Runs a counter thread. '''
    countChanged = pyqtSignal(int)
    count = 0

    def run(self):
        while self.count < TIME_LIMIT:
            self.count +=1
            time.sleep(0.01)
            self.countChanged.emit(self.count)

    def initCount(self, num):
        self.countChanged.emit(num)
        self.count = num


class InThread(threading.Thread):
    def __init__(self, threadID, eit: Eit, mtd):
        global ID_SUM, ID, exitFlag, Que, threads, queueLock

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.id = threadID
        self.eit = eit
        self.mtd = mtd
        self.data = {}
        self.ready = False

    def run(self):
        global ID_SUM, ID, exitFlag, Que, threads, queueLock
        while not exitFlag:
            queueLock.acquire()
            Que.put(self.eit.solver(self.mtd, self.id))
            print('{} solved the problem {}'.format(self.threadID, self.id))
            self.id = ID_SUM
            ID_SUM += 1
            ID += 1
            queueLock.release()
            
            time.sleep(0.1)

                
class OutThread(threading.Thread):
    def __init__(self, threadID, eit: Eit, canvas):
        global ID_SUM, ID, exitFlag, Que, threads, queueLock

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.canvas = canvas
        self.eit = eit
        self.data_list = []

    def run(self):
        global ID_SUM, ID, exitFlag, Que, threads, queueLock
        while not exitFlag:
            print('Try to plot')
            queueLock.acquire()
            while not Que.empty():
                data = Que.get()
                self.data_list.append(data)
            queueLock.release()

            # ! plot
            for data in  self.data_list:
                self.canvas.ax.cla() 
                print('{} ploted the fig {}'.format(self.threadID, data['id']))
                self.eit.plot(ax=self.canvas.ax, data = data)
                self.canvas.draw()
                time.sleep(0.2)

            # ! fixed
            time.sleep(0.2)

class offlineThread(threading.Thread):
    def __init__(self, threadID, eit: Eit, canvas, fpath):
        global ID_SUM, ID, exitFlag, Que, threads, queueLock

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.canvas = canvas
        self.eit = eit
        self.data_list = np.load(fpath, allow_pickle=True).item(0)['data']

    def run(self):
        global ID_SUM, ID, exitFlag, Que, threads, queueLock
        while not exitFlag:
            # ! plot
            for data in self.data_list:
                self.canvas.ax.cla() 
                print('{} ploted the fig {}'.format(self.threadID, data['id']))
                self.eit.plot(ax=self.canvas.ax, data = data)
                self.canvas.draw()
                time.sleep(0.03)

def distrib(thread_num, eit, mtd, canvas, offline, fpath):
    # create in threads
    global ID_SUM, ID, exitFlag, Que, threads, queueLock

    if not offline:
        ID_SUM, ID = thread_num, 0
        for threadID in range(thread_num):
            thread = InThread(threadID, copy.copy(eit), mtd)
            thread.start()
            threads.append(thread)

        # create out threads
        thread = OutThread(thread_num, eit, canvas)
        thread.start()
        threads.append(thread)
    else:
        thread = offlineThread(0, eit, canvas, fpath)
        thread.start()
        threads.append(thread)

def join():
    global ID_SUM, ID, exitFlag, Que, threads, queueLock, save_data_list
    if len(save_data_list):
        save_data_list = threads[-1].data_list

    try:
        for t in threads:
            t.join()
        threads.clear()
    except:
        print('Errors occupied when relase all th threads.')

def save_mat(fpath):
    global save_data_list
    if len(threads):
        np.save(fpath, {'data': threads[-1].data_list})
        return 'save sucess!'
    elif len(save_data_list):
        np.save(fpath, {'data': save_data_list})
        return 'save sucess!'
    return 'thread is empty!'