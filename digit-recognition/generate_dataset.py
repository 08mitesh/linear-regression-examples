import glob
import cv2
import csv


label_list = ["0","1","2","3"]

for label in range(len(label_list)):

	str_label = str(label)
	dirList = glob.glob("C:\\Users\\Admin\\Downloads\\digit recog new\\digit recog\\orig_images\\"+str_label+"\\*.png")

	file_path = open("digit_recognizer_dataset.csv","a",newline='')
	writer = csv.writer(file_path)

	if label == 0:
		header = ["label"]
		for x in range(784):
			header.append("Pixel_"+str(x))
		writer.writerow(header)

	for img_path in dirList:

		## Read image file using openCV
		img = cv2.imread(img_path,0) # reads image  as grayscale

		img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)

		cv2.imshow("Testing",img)

		data = [] 

		data.append(label)

		rows,cols = img.shape

		# print(rows,cols)

		for i in range(rows):
			for j in range(cols):
				value = img[i,j]

				if value < 100:
					value = 0
				else:
					value = 1

				data.append(value)
		
		# print(data)
		writer.writerow(data)
	# break

	# cv2.waitKey()

	# break
