def BIO_to_BIO2(file_path=r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03', file_name = 'ner_deu.dev', tag_pos = 3):
    with open(file_path+'\\'+file_name,'r') as read_file:
        with open(file_path+'\\'+file_name.replace('.','_IBO2.'),'w') as new_file:
            original = read_file.read().split('\n')
            original = [column.split(' ') for column in original]

            last_tag = ''
            for word_line in original:
                if word_line[0] != '':
                    tag = word_line[tag_pos]

                    if (tag == 'I-LOC' or tag == 'I-ORG' or tag == 'I-PER' or tag == 'I-MISC') and last_tag == 'O':
                        new_tag = tag.replace('I','B',1)
                        word_line[tag_pos] = new_tag
                        new_file.write(' '.join(word_line)+'\n')
                        last_tag = tag
                    else:
                        new_file.write(' '.join(word_line)+'\n')
                        last_tag = tag

                else:
                    new_file.write('\n')