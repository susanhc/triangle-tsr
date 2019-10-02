with open('train.txt') as f:
	contents=f.readlines()
	for content in contents:
		content=content.strip('\n').split(';')
		width=content[1]
		print(width)