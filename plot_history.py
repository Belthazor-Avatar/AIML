def plot_history(hist):
    fig, ax = plt.subplots(2,1, figsize=(16,10))
    ax[0].plot(hist.history['loss'])
    ax[0].plot(hist.history['val_loss'])
    ax[0].legend(['loss', 'val_loss'])
    ax[0].title.set_text('Train loss vs Val loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')

    ax[1].plot(hist.history['accuracy'])
    ax[1].plot(hist.history['val_accuracy'])
    ax[1].legend(['accuracy', 'val_accuracy'])
    ax[1].title.set_text('Train acc vs Val Acc')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epochs')

    ax[0].grid()
    ax[1].grid()

    print(f"Max Training Accuracy {max(hist.history['accuracy'])} \nMax Val Accuracy {max(hist.history['val_accuracy'])}")
