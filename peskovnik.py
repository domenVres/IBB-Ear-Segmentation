class Stars:
    def __init__(self, parameter):
        self.parameter = parameter
        print("Kreiram starsa")

    def izpisi(self):
        print(self.parameter)

    def __getitem__(self, item):
        return "Stars" + str(item)

class Otrok(Stars):
    def __init__(self, parameter):
        super().__init__(parameter)
        self.parameter = parameter + "Otrok"

    def __getitem__(self, item):
        i = super().__getitem__(item)
        return "Otrok" + i

if __name__=="__main__":
    o = Otrok("Test")
    o.izpisi()
    print(o[2])
