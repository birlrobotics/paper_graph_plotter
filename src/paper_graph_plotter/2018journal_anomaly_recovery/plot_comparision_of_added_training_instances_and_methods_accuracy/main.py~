import xlrd
import unicodecsv
import os
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

def xls2csv (xls_filename, csv_filename):
    # Converts an Excel file to a CSV file.
    # If the excel file has multiple worksheets, only the first worksheet is converted.
    # Uses unicodecsv, so it will handle Unicode characters.
    # Uses a recent version of xlrd, so it should handle old .xls and new .xlsx equally well.

    wb = xlrd.open_workbook(xls_filename)
    sh = wb.sheet_by_index(0)

    fh = open(csv_filename,"wb")
    csv_out = unicodecsv.writer(fh, encoding='utf-8')

    for row_number in xrange (sh.nrows):
        csv_out.writerow(sh.row_values(row_number))

    fh.close()

def run(csv_filename):
    if csv_filename is None:
        print 'Worng *.csv file csv_filename'
        os.exit()

    df = pd.read_csv(csv_filename)
    list_of_df = []
    list_of_method_config = []
    for icase in range(8):
        ind = range(icase*7, (icase+1)*7)
        method_config = ''.join('{} '.format(val) for val in df.loc[ind[0], df.columns[0:3]].values)
        list_of_method_config.append(method_config)
        idf = df.loc[df.index[ind]]
        idf = idf.sort_values(by = 'train_size', ascending = True)
        list_of_df.append(idf)

    #plot instances_vs_accuracy
    fig, ax = plt.subplots(nrows =1, ncols = 1, figsize=(12,6))
    for i, icof in enumerate(list_of_method_config):
        arr = list_of_df[i][['train_size', 'accuracy']].values
        if i == 0:
            xs = arr[:, 0]
        ax.plot(xs, arr[:,1], marker = 'o', label=icof)
    ax.legend()
    ax.set_title('Trade-off between the number of training instances and models configurations')
    ax.set_xlabel('Training instances added')
    ax.set_ylabel('Average accuracy ')
    fig.savefig('instances_vs_accuracy.eps', dpi=300)

    #plot instances_vs_F1score
    fig, ax = plt.subplots(nrows =1, ncols = 1, figsize=(12,6))
    for i, icof in enumerate(list_of_method_config):
        arr = list_of_df[i][['train_size', 'F1score']].values
        if i == 0:
            xs = arr[:, 0]
        ax.plot(xs, arr[:,1], marker = 's', label=icof)
    ax.legend()
    ax.set_title('Trade-off between the number of training instances and models configurations')
    ax.set_xlabel('Training instances added')
    ax.set_ylabel('Average F1score ')
    fig.savefig('instances_vs_F1score.eps', dpi=300)

    plt.show()
        
        
if __name__=="__main__":
    xls_filename = 'trade_off_no_soVB.xlsx'
    csv_filename = 'trade_off_no_soVB.csv'
    xls2csv(xls_filename, csv_filename)
    run(csv_filename=csv_filename)
