from abc import ABC, ABCMeta, abstractmethod


class test_class(ABC):

    # instant
    def test_method(self):
        print('class instant')
        return None

    # class method
    @classmethod
    def test_class_method(cls):
        print('class class')
        return None

    # static method
    @staticmethod
    def test_static_method():
        print('class static')
        return None

    # abstract method
    @abstractmethod
    def _abs_method(self):
        print('class abs method')
        pass


class test_subclass(test_class):
    def test_method(self):
        print('subclass instant')
        return None


temp = test_subclass()
temp.test_static_method()
temp._abs_method()
