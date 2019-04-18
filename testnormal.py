import _thread

def function():
	count = 0
	while count < 5:
		print(count)
		count += 1

if __name__ == '__main__':
	_thread.start_new_thread(function, ())
	while 1:
		pass

