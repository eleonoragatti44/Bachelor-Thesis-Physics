loader = torch.utils.data.DataLoader(dataset,
									 batch_size = 126,
									 shuffle = True)
train, validation = loader