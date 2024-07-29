def confusion_matrix_heatmap(cmh_df, cmh_labels):
    # Generate confusion matrix
    cm = confusion_matrix(cmh_df.y_test, cmh_df.pred)
    # Draw heatmap
    plt.figure(figsize=(8,8))
    plt.tick_params(axis='x', labelrotation=90)
    sns.heatmap(cm, xticklabels=cmh_labels.keys(), yticklabels=cmh_labels.keys(), annot=True, cmap='mako', center=0, fmt='d')
    plt.show()


