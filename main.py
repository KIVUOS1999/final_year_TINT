import functions as f

X = f.load_file()
X = X/255.0

#if new model uncomment

discriminator = f.discriminator()
generator = f.generator(100)
gan = f.gan(discriminator, generator)
'''

#if training already existed model

discriminator = f.load_model('discriminator.model')
generator = f.load_model('generator.model')
gan = f.load_model('gan.model')


random_noise = f.np.random.randn(100).reshape(1,100)
gen_image = generator.predict(random_noise)
'''

itternations = 100
batch_size = 256
n = 100 

for i in range(1, itternations+1):
    for j in range(len(X)//batch_size):
        x_real = X[f.np.random.randint(0, len(X), batch_size//2)].reshape(batch_size//2, 64,64,1)
        y_real = f.np.ones(batch_size//2).reshape(batch_size//2, 1)

        x_fake = generator(f.np.random.randn(batch_size//2, n))
        y_fake = f.np.zeros(batch_size//2).reshape(batch_size//2, 1)

        combined_x = f.np.vstack((x_real, x_fake))
        combined_y = f.np.vstack((y_real, y_fake))

        dloss = discriminator.train_on_batch(combined_x, combined_y)
        gloss = gan.train_on_batch(f.np.random.randn(batch_size, n), f.np.ones(batch_size).reshape(batch_size, 1))

        print("Current:", "EPOCH -->", i,"/",itternations, "Batch-->",j,'/',batch_size,'discrim loss',dloss,'gan loss',gloss)
    
    if i%50 == 0:
        discriminator.save('discriminator.model')
        gan.save('gan.model')
        generator.save('generator.model')
    
    f.preview(i, generator) 

'''
#This code is to fenerate the set of original images which we are going to achieve
fig, axis = f.plt.subplots(10,10, figsize = (20,20))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(
            X[i+j], cmap = 'gray'
        )
f.plt.savefig("temp/original.png")
f.plt.close()

'''