# Xử lý Text để speech
Phat_am = {
    'II':'hai', 'III':' ba', ' IV':' bốn', 'VI':'sáu', 'VII':'bảy', 'VIII':'tám', 'IX':'chín', '-':' ',
    ' VCS':' vi si ét', ' vcs':' vi si ét ', '/QĐ-TTg':' theo quyết định của thủ tướng chính phủ', '/':' gạch ',
    # Đọc phần mềm 
    'ATS.NET':'ây ti ét chấm nét', 'RMS.NET':'a em ét chấm nét', 'PRM.NET':'pi a em chấm nét', 'GFS.NET':'gi ép ét chấm nét',
    
    # 
    'KT-XH':'kinh tế xã hội', 'DTTS':'dân tộc thiểu số',
    
    
    # Viết tắt
    '/QH':' theo quốc hội ', '/CP':' theo chính phủ ', '/NĐ-CP': ' theo nghị định của chính phủ', '/NQ':' theo nghị quyết ',
    '/TT-BTC':' theo thông tư bộ tài chính', 
}
KHLoaiVB = ['/NĐ-CP', '/QH14', '/QĐ-TTg', '/TT-BNNPTNT', '/QĐ-BCĐCTMTQG', '/TT-UBDT', '/QĐ-BGDĐT', '/QH15', '/TT-BTTTT', '/QĐ-UBDT', '/NQ-UBTVQH15', '/VBHN-BKHĐT', '/TT-BTC', '/TT-BXD', '/QH13', '/NQ-CP', '/TT-BLĐTBXH']

# Sắp xếp từ lớn đến bé tránh trường hợp thay thế chuỗi con
Phat_am = dict(sorted(list(Phat_am.items()), key=lambda item: len(item[0].strip()), reverse=True))


# Xử lý Text từ audio để in ra màn hình

Dich_am = {
    'gạch ngang':'-', 'gạch chéo':' / ', ' xược ':'/', ' gạch ':'/', 'theo quốc hội số ':'QH', ' của quốc hội số ':'QH',
    'theo nghị định của chính phủ ':'NĐCP', 'nợ tài khoản':'\nNợ TK:', 'có tài khoản':'\n______Có TK:',
    ' phẩy':',', ' 2 chấm': ':', ' hai chấm':':', ' xuống dòng ': '\n', 'hồ chí minh': 'Hồ Chí Minh', 'việt nam':'Việt Nam',
    'hà nội':'Hà Nội',
    'hê lô':'hello', 'thê lô':'hello', 'a lô': 'alo',
    'vi si ét':'VCS', 'vê sê ét':'VCS', 'vi si ết':'VCS','vi sê ết':'vcs', 
}

Chu_cai = {
    'a':'a', 'bê':'b', 'bi':'b', 'sê':'c', 'si':'c', 'đê':'d', 'e':'e', 'ep':'f', 'gờ':'g',
    'hat':'h', 'hát':'h', 'i':'i', 'zi':'j', 'gi':'j', 'ca':'k', 'lờ':'l', 'mờ':'m', 'nờ':'n', 'o':'o',
    'pê':'p','te':'t', 'qui':'q', 'rờ':'r', 'ét':'s', 'tê':'t', 'u':'u', 'vê kép':'w',
}

number = { 'không': 0,'một': 1,'hai': 2,'ba': 3,'bốn': 4,'năm': 5, 'lăm':5,'sáu': 6,'bảy': 7, 'bẩy':7,'tám': 8,'chín': 9,'mười': 10, 
          'linh':0,'mốt':1,'tư':4,}
number_dac_biet = ['linh','mốt','tư']
donvi = {'tỷ': 1000000000, 'vạn':10000000, 'triệu': 1000000, 'nghìn': 1000,'trăm': 100, 'mươi': 10}
catch_year = ['vào', 'trong', 'đầu', 'cuối', 'giữa', 'suốt', 'từ', 'vượt', 'qua', 'hết', 'sang', 'cả', 'mỗi', 'theo', 'hàng', 'đến']

# Sắp xếp từ lớn đến bé tránh trường hợp thay thế chuỗi con
Dich_am = dict(sorted(list(Dich_am.items()), key=lambda item: len(item[0].strip()), reverse=True))
Chu_cai = dict(sorted(list(Chu_cai.items()), key=lambda item: len(item[0].strip()), reverse=True))


def Xu_ly_text_de_doc(text:str):
    for v in ['.',',','.',':',';',' ']:
        text = text.replace(' I'+v,'một,').replace(' V'+v,'năm,').replace(' X'+v,'mười,')
    for v in KHLoaiVB:
         text = text.replace(v,'')
    text = text.replace(" (",", ").replace(")",".")
    for old, new in Phat_am.items():
        text = text.replace(old, new)
    return text


# Hàm nội bộ
def Check_dvi_giam_dan_hoac_chi_1dvi(lst:list):
    dvi_truoc = None
    for v in lst:
        if v in donvi:
            if dvi_truoc == None: # nếu donvi hiện tại < donvi trước thì tiếp tục duyệt tìm 
                dvi_truoc = donvi[v]
            elif donvi[v] < dvi_truoc:
                dvi_truoc = donvi[v]
            else: # Nếu donvi hiện tại > donvi trước return False
                return False
    return True
def Tinh_dvi_giam_dan(lst:list):
    total = 0
    num = None
    for v in lst:
        if num != None and v in donvi:
            total += num*donvi[v]
            num = None
        elif num != None and v in number:
            if number[v] == 10:
                num = num*100 + number[v]
            elif num_truoc == 10:
                num += number[v]
            elif num_truoc == 0:
                num = num*10 + number[v]
            else:
                num = num*10 + number[v]
            num_truoc = number[v]
        else:
            num = number[v]
            num_truoc = number[v]
        
    if num != None: total += num
    return total
def Check_so_dvi_la_1(lst:list):
    dem = 0
    for v in lst:
        if v in donvi: v += 1
        if v > 1: return False
    return True
def Tim_chi_so_dvi_lon_nhat(lst:list):
    max = 0
    index_max = None
    for i in range(len(lst)):
        if lst[i] in donvi:
            if donvi[lst[i]] > max:
                max = donvi[lst[i]]
                index_max = i
    return index_max, max
def Digit_to_Number(lst:list):
    if(Check_dvi_giam_dan_hoac_chi_1dvi(lst)):
        return Tinh_dvi_giam_dan(lst)
    else:
        idx, dvi_max = Tim_chi_so_dvi_lon_nhat(lst)
        return Digit_to_Number(lst[:idx])*dvi_max + Digit_to_Number(lst[idx+1:])
def De_quy_xu_ly_so(lst:list)->str:
    # Xử lý trường hợp số 0 ở đầu
    dem_0 = 0
    for v in lst:
        if v == 'không': dem_0 += 1
        else: break
    if dem_0 == len(lst): return dem_0*'0'
    return dem_0*'0' + str(Digit_to_Number(lst))
def Loc_lst_num_va_sap_Xep(words:str)->list:
    lst_num = []
    num = []
    is_num = False
    for v in words.split(' '):
        if v in number_dac_biet and not num: continue
        if v in number or v in donvi:
            if v in number: 
                num += [v]
                is_num = True
            if v in donvi and is_num:
                num += [v]
        elif len(num)>=1:
            if len(num)==1 and num[0] in ['ba','chín']: # Do dùng replace nên tránh TH chữ ví dụ 'bao' -> '3o'
                num = []
            else:
                lst_num += [num]
                num = []
                is_num = False
        else: num = []
    if len(num)>=1: lst_num += [num]
    lst_num = sorted(lst_num, reverse=True)
    return lst_num
def Xu_ly_so_cho_text(words:str)->str:
    lst_num = Loc_lst_num_va_sap_Xep(words)
    dic_xuly_num = {}
    for v in lst_num:
        dic_xuly_num[' '.join(v)] = De_quy_xu_ly_so(v)
    dic_xuly_num = dict(sorted(list(dic_xuly_num.items()), key=lambda item: len(item[0].strip()), reverse=True))
    for old,new in dic_xuly_num.items():
        words = words.replace(old,new)
    return words
def Xu_ly_DMY_cho_text(words:str):
    lst = []
    dmy:list[str] = []
    check = False
    thutu_dmy = {
        'ngày': 'tháng', 'tháng':'năm', 'năm':'ngày', 'mùng':'tháng'
    }
    dmy_ketiep = ''
    for v in words.split(' '):
        # Xử lý các trường hợp gặp từ ngày tháng năm
        if v in thutu_dmy and check == False:
            if (not dmy) or (dmy and v == dmy_ketiep):
                dmy.append(v)
                dmy_ketiep = thutu_dmy[v]
            elif len(dmy) >= 4: 
                lst.append(dmy)
                dmy = [v]
            else: dmy = [v]
            check = True
        # Xử lý các trường hợp gặp số mà trước đó có từ ngày tháng năm
        elif v.isdigit() and check == True:
            dmy.append(v)
            check = False
            if len(dmy) >= 4:
                if dmy[-2] == 'năm':
                    lst.append(dmy)
                    dmy = []
        # Xử lý các ngoại lệ
        elif len(dmy)>=4:
            lst.append(dmy)
            dmy = []
        else: dmy = []

    # return lst
    lst = sorted(lst,reverse=True)
    lst2  = [' '.join(v) for v in lst]
    result = {}
    for v in range(len(lst)):
        for i in range(2,len(lst[v]),2):
            lst[v][i] = '/'
        result[lst2[v]] =  lst[v][0] + ' ' + ''.join(lst[v][1:])
    result = dict(sorted(list(result.items()), key=lambda item: len(item[0].strip()), reverse=True))
    for old, new in result.items():
        words = words.replace(old,new)
    return words
     
# Hàm toàn cục 

def Xu_ly_text(words:str)->str:
    if words == '': return ''
    # Xử lý số trước
    words = Xu_ly_so_cho_text(words)
    
    # Xử lý các từ đặc biệt
    for old,new in Dich_am.items():
        words = words.replace(old,new)
    # Xử lý các ý gạch đầu dòng 
    for old,new in Chu_cai.items():
        words = words.replace('í '+old,'\n'+new+')')
        words = words.replace('ý '+old,'\n'+new+')')
        words = words.replace(old+' nhỏ','\n'+new+')')
        words = words.replace(old+' lớn','\n'+new.upper()+')')
    
    # Xử lý chữ 'năm'
    if words[0] == '5': words = 'năm '+words[1:]
    for i in range(len(words)):
        try:
            if words[i] == '5' and (words[i-8:i-3]=='tháng' or words[i-7:i-2]=='tháng'):
                words = words[:i] + ' năm ' + words[i+1:]
        except: continue
    for v in catch_year:
        val = v + ' 5'
        if val in words:
            words = words.replace(val, v+' năm ')
    # Xử lý ngày tháng năm 
    words = Xu_ly_DMY_cho_text(words)

    # Xử lý số ba và số chín:
    for v in [' ',',','.',';']:
        words = words.replace('ba'+v,'3'+v)
        words = words.replace('chín'+v,'9'+v)


    # Viết hoa chữ cái đầu 
    if len(words)>1:
        words = words[0].upper()+words[1:]+'.'
    return words
