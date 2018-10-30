class Pizza(object):
    @staticmethod
    def mix_ingredients(x, y):
        return x + y
    def cook(self):
        print(self.mix_ingredients(1,3))

print(Pizza.mix_ingredients(1,3))
Pizza().cook()
