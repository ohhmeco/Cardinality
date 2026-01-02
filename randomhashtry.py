import randomhash

# Create a family of random hash functions, with 10 hash functions

rfh = randomhash.RandomHashFamily(count=10)
print(rfh.hashes("hello"))  # will compute the ten hashes for "hello"
print(rfh.hash("hello"))
#print(randomhash.hashes("hello", count=10))
