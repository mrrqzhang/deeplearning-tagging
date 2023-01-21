import sys,csv,re
from collections import defaultdict
import math

csv.field_size_limit(sys.maxsize)

excluded_set = [ "[后_brand]", "[24小时_brand]", "[cp]", "[/cp]" ]
def calc_log(x):
    if x<=0: return 0
    else: return math.log(x)

def calc_info(a, b, c):
    if a==0 or b==0 or c==0: return 0
    else: return calc_log(a/b/c)



def calc_mi(px,py,pxy):
    pnx = 1-px
    pny = 1-py
    pnx_y = py - pxy
    px_ny = px - pxy
    pnx_ny = 1 - pxy - pnx_y - px_ny
    mi = pxy*calc_info(pxy,px,py) + pnx_y*calc_info(pnx_y, pnx, py) + px_ny*calc_info(px_ny,px, pny) + pnx_ny*calc_info(pnx_ny,pnx,pny)
    return mi

def main():
    filename="/data0/congmin/data/query_text_9_days.sel.ne.csv"
    nedict={}
    with open(filename, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            mid = row['mid']
            text =row['text']
            nes = re.findall('\[.*?\]', text)
            if nes:
                nes_clean = [ ne for ne in nes if ne not in excluded_set ]
                if mid not in nedict: nedict[mid] = set(nes_clean)

    filename="/data0/congmin/data/query_text_9_days.sel.tag.csv"
    cntx = defaultdict(int)
    cnty = defaultdict(int)
    cntxy = defaultdict(int)
    nsample = 0
    with open(filename, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            mid = row['mid']
            tag = row['keywords']
            query = row['text']
            if mid not in nedict: continue
            if tag=='': continue
            nsample+=1
            ys = nedict[mid]
            for x in tag.split(): cntx[x]+=1
            for y in ys: cnty[y] += 1
            for x in tag.split():
                for y in ys:
                    cntxy[x+'\t'+y] += 1

    cntmi = {}
    for key in cntxy:
        x,y = key.split('\t')
        nsample = cntx[x] + cnty[y]   # use only x and y occur cases
        px = cntx[x]*1.0/nsample
        py = cnty[y]*1.0/nsample
        pxy = cntxy[key]*1.0/nsample
        cntmi[key] = calc_mi(px,py,pxy)

    for key in sorted(cntxy.keys()):
        x,y = key.split('\t')
        print(key,"\t",cntmi[key], "\t", cntx[x], "\t", cnty[y], "\t", cntxy[key] )




if __name__== "__main__":
  main()

