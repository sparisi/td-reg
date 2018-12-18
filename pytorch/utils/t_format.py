
def t_format(text, text_length=0):
    if text_length==0:
        return "%-10s" % text 
    elif text_length==1:
        return "%-20s" % text
    elif text_length==2:
        return "%-25s" % text
    else:
        return "%-25s" % text

