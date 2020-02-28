import numpy as np

def write_str(w,str):
	w.write(len(str).to_bytes(4, byteorder='big'))
	w.write(str.encode())

def read_str(w):
	str_len = int.from_bytes(w.read(4), byteorder='big')
	str = w.read(str_len).decode()
	return str

def write_array(w,array):
	array_shape = np.array(array.shape).astype('i').tostring()
	array_bytes = array.astype('f').tostring()
	w.write(len(array_shape).to_bytes(4, byteorder='big'))
	w.write(array_shape)
	w.write(len(array_bytes).to_bytes(4, byteorder='big'))
	w.write(array_bytes)

def read_array(w):
	array_bytes = None
	array_shape_len = int.from_bytes(w.read(4), byteorder='big')
	array_shape = np.fromstring(w.read(array_shape_len),dtype=np.int32)
	array_len = int.from_bytes(w.read(4), byteorder='big')
	array = np.fromstring(w.read(array_len),dtype=np.float32)
	array = array.reshape(array_shape)
	return array

def save(file,net_description,variables):
	with open(file, 'wb') as w:
		write_str(w,"KCIRE NN SAVEFILE,V1.0")
		write_str(w,net_description)
		w.write(len(variables).to_bytes(4, byteorder='big'))
		for key, value in variables.items():
			write_str(w,key)
			write_array(w,value)

def load(file):
	variables = {}
	print("loading ",file)
	with open(file, 'rb') as w:
		file_version = read_str(w)
		net_description = read_str(w)
		print(file_version)
		print(net_description)
		dict_len = int.from_bytes(w.read(4), byteorder='big')
		for i in range(dict_len):
			key = read_str(w)
			value = read_array(w)
			variables[key] = value
			print(key,":",value.shape)
	return variables

if __name__ == '__main__':
	t1 = np.random.uniform(0.0,1.0,10).reshape(5,2)

	w1 = np.random.uniform(0.0,1.0,100).reshape(5,5,1,4)

	w2 = np.random.uniform(0.0,1.0,800).reshape(5,5,4,8)

	w3 = np.random.uniform(0.0,1.0,25088).reshape(7*7*8,64)

	w4 = np.random.uniform(0.0,1.0,640).reshape(64,10)

	#t4 = np.random.uniform(0.0,1.0,100000000).reshape(1000,10,10,10,10,10)

	variables = {"t1":t1,"w1":w1,"w2":w2,"w3":w3,"w4":w4}
	print("saved:")
	for key, value in variables.items():
		print(key,":",value.shape)

	save("w1.test","Testando apenas",variables)

	loaded = load("w1.test")
	print("loaded:")
	for key, value in loaded.items():
		print(key,":",value.shape)

	#print("shape:",loaded.shape)