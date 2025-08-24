
matrix = (
    (4, 10, 9, 2, 13, 8, 0, 14, 6, 11, 1, 12, 7, 15, 5, 3),
    (14, 11, 4, 12, 6, 13, 15, 10, 2, 3, 8, 1, 0, 7, 5, 9),
    (5, 8, 1, 13, 10, 3, 4, 2, 14, 15, 12, 7, 6, 0, 9, 11),
    (7, 13, 10, 1, 0, 8, 9, 15, 14, 4, 6, 12, 11, 2, 5, 3),
    (6, 12, 7, 1, 5, 15, 13, 8, 4, 10, 9, 14, 0, 3, 11, 2),
    (4, 11, 10, 0, 7, 2, 1, 13, 3, 6, 8, 5, 9, 12, 15, 14),
    (13, 11, 4, 1, 3, 15, 5, 9, 0, 10, 14, 7, 6, 8, 2, 12),
    (1, 15, 13, 0, 5, 7, 10, 4, 9, 2, 3, 14, 6, 11, 8, 12),
)

def get_out(inright, key):
    out = 0
    temp = (inright + key) % (1 << 32)
    # Сдвиг суммы ключа и правой позиции
    for i in range(8):
        phonetic = (temp >> (4 * i)) & 0b1111
        '''
        Определение ячейки
        Сдвиг на n бит и выбор пересечения
        '''
        out |= (matrix[i][phonetic] << (4 * i))
        '''
        Второй бит устанавливается в 0
        Сдвиг влево на n бит
        '''
    out = ((out >> 21) | (out << 11)) & 0xFFFFFFFF
    # Пересечение бита и объединения множеств со сдвигами
    return out
def crypt_operation(inleft, inright, key):
    outleft = inright
    # Присваиваем левому выходу правые входные данные
    outright = inleft ^ get_out(inright, key)
    # Исключаем пересечение
    return outleft, outright

class Gost:
    def __init__(self,num_rounds=32):
        self.num_rounds=num_rounds
        self.key = [None] * 8

    def set_key(self, key):
        for i in range(8):
            self.key[i] = (key >> (32 * i)) & 0xFFFFFFFF

    def crypt(self, text):
        #text = int(text.encode('utf-8').hex(), 16)

        text_left = text >> 32
        text_right = text & 0xFFFFFFFF
        # Разбитие текста с помощью побитового сдвига
        n_r=0
        for q in range(24):
            if n_r==self.num_rounds:
                break
            text_left, text_right = crypt_operation(text_left, text_right, self.key[q % 8])
            n_r+=1
        for q in range(8):
            if n_r==self.num_rounds:
                break
            text_left, text_right = crypt_operation(text_left, text_right, self.key[7 - q])
            n_r+=1
        # 32 цикла с различными ключами
        hash = (text_left << 32) | text_right
        return hash
