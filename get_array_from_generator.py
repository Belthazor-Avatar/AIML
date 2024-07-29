def get_array_from_gen(generator):
    X = []
    y = []
    generator.reset()
    for i in range(generator.__len__()):
        try:
            a, b = next(generator)
            X.append(a)
            y.append(b)
        except OSError as e:
            print(f"Skipping this item due to an error: {str(e)}")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print(X.shape)
    print(y.shape)
    return X, y


