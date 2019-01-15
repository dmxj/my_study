
class _DataIterator():
    def __init__(self,data_loader):
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.i = 0
        self.cnt = self.__len__()

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        while True:
            item = "data item:{}".format(self.dataset[self.i%self.cnt])
            self.i += 1
            return item



class DataLoader():
    def __init__(self,dataset):
        self.dataset = dataset

    def __iter__(self):
        return _DataIterator(self)

ds = list(range(10))
dl = DataLoader(ds)
for i,v in enumerate(dl):
    print(i," ==> ", v)

