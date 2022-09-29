import time 
import numpy as np
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import (
    ROOT, 
    split_dataset, 
    create_triplets, 
    get_siamese_network, 
    SiameseModel, 
    get_batch, 
    test_on_triplets,
    extract_encoder,
    classify_images,
)

def run():
    train_list, test_list = split_dataset(ROOT, split = 0.9)
    print("Length of training list:", len(train_list))
    print("Length of testing list :", len(test_list))

    # train_list, test list contains the folder names along with the number of files in the folder.
    print("\nTest List:", test_list)

    train_triplet = create_triplets(ROOT, train_list)
    test_triplet  = create_triplets(ROOT, test_list)

    print("Number of training triplets:", len(train_triplet))
    print("Number of testing triplets :", len(test_triplet))

    print("\nExamples of triplets:")
    for i in range(5):
        print(train_triplet[i])

    siamese_network = get_siamese_network()
    print(siamese_network.summary())

    siamese_model = SiameseModel(siamese_network)
    optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
    siamese_model.compile(optimizer=optimizer)

    save_all = False
    epochs = 256
    batch_size = 128

    max_acc = 0
    train_loss = []
    test_metrics = []

    for epoch in tqdm(range(1, epochs+1), desc = "Epoch"):
        t = time.time()
        
        # Training the model on train data
        epoch_loss = []
        for data in get_batch(train_triplet, batch_size=batch_size):
            loss = siamese_model.train_on_batch(data)
            epoch_loss.append(loss)
        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        train_loss.append(epoch_loss)

        print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time()-t)} sec)")
        print(f"Loss on train    = {epoch_loss:.5f}")
        
        # NOTE: check test_triplets.
        # print(test_triplet)

        # Testing the model on test data
        metric = test_on_triplets(batch_size = batch_size, test_triplet = test_triplet, siamese_model = siamese_model)
        test_metrics.append(metric)
        accuracy = metric[0]
        
        # Saving the model weights
        if save_all or accuracy >= max_acc:
            siamese_model.save_weights("siamese_model")
            max_acc = accuracy
            if accuracy >= 0.9:
                break
        
    # Saving the model after all epochs run
    # siamese_model.save("siamese_model_all.h5")
    siamese_model.save_weights("siamese_model-final")

    # Save the model of encoder extractor
    encoder = extract_encoder(siamese_model)
    # encoder.save("encoder_all.h5")
    encoder.save_weights("encoder")
    print(encoder.summary())

    pos_list = np.array([])
    neg_list = np.array([])

    for data in get_batch(test_triplet, batch_size=256):
        a, p, n = data
        pos_list = np.append(pos_list, classify_images(a, p, encoder = encoder))
        neg_list = np.append(neg_list, classify_images(a, n, encoder = encoder))
        break

    # Compute and print the accuracy
    true = np.array([0]*len(pos_list)+[1]*len(neg_list))
    pred = np.append(pos_list, neg_list)
    
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}")

if __name__ == "__main__":
    run()