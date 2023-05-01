






def refactor_pl():
    # posting_file_loc = "/localdisk3/data-selection/data/metadata/fashion-mnist/0.9/resnet-18.txt"
    posting_file_loc = '/localdisk2/fkc_coreset_cpp_results/fashion_mnist_pl-resnet-18_0.9.txt'
    posting_file_loc_rf = "/localdisk2/fkc_coreset_cpp_results/fashion-mnist_resnet-18_rf.txt"

    posting_list_file = open(posting_file_loc, 'r')
    lines = posting_list_file.readlines()
    with open(posting_file_loc_rf, 'w') as out:
        for line in lines:
            pl = line.split(':')
            key = int(pl[0])
            value = pl[1].split(',')
            value = [v.replace("{", "").replace("}", "").strip() for v in value]
            # out_string = ' '.join(value) + '\n'
            # print(out_string)
            # break
            out.write(' '.join(value) + '\n')
    posting_list_file.close()



if __name__ == "__main__":
    refactor_pl()