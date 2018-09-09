import pylab
import imageio
filename = '/tmp/file.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
nums = [10, 287]
for num in nums:
    image = vid.get_data(num)
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(num), fontsize=20)
    pylab.imshow(image)
pylab.show()