from transliterate import translit

with open('darn.txt') as f:
    content = f.readlines()
print content
darn = [translit(x.encode(),"ru", reversed=True) for x in content]
print darn
