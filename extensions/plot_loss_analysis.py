

def loss_analysis(fname):
    lines = read_file(fname)
    disc_losses = []
    evals = []
    validations = []

    for line in lines:
        if 'Discriminator loss: ' in line:
            disc_losses.append(float(line.split('Discriminator loss:')[1].split('-')[0]))
        elif 'csls_knn_10 - Precision at k = 1:' in line:
            evals.append(float(line.split('csls_knn_10 - Precision at k = 1:')[1].strip()))
        elif 'Mean cosine (csls_knn_10 method, S2T build, 10000 max size):' in line:
            validations.append(float(line.split(':')[-1].strip()))
        elif 'src_emb' in line:
            src_emb = line.split('src_emb:')[-1].strip().split('/')[-2]
        elif 'tgt_emb' in line:
            tgt_emb = line.split('src_emb:')[-1].strip().split('/')[-2]
        elif 'normalize_embeddings' in line:
            norm = line.split('normalize_embeddings:')[-1].strip()
    print(disc_losses)
    print(evals)
    print(validations)
    x_s = [i*0.1 for i in range(len(disc_losses))]

    handle1 = plt.plot(x_s, disc_losses, color='g')
    handle2 = plt.plot(x_s, [elm*0.01 for elm in evals], color='b')
    handle3 = plt.plot(x_s, validations, color='r')
    plt.legend(handles=[handle1[0], handle2[0], handle3[0]],
               labels=['Discriminator loss', 'P@1', 'Mean cosine (csls_knn_10)'])
    name_str = '{}_{}'.format(src_emb, tgt_emb)
    if norm != '':
        name_str += '_norm'
    else:
        name_str += '_nonnorm'
    name_stem = name_str
    name_str += ' ({})'.format(fname.split('/')[-2])
    plt.title(name_str)
    #plt.show()
    plt.savefig('logs/plots/{}.pdf'.format(name_stem), bbox_inches='tight')
    plt.clf()