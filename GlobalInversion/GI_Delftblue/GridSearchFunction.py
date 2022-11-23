def gridsearch(Zdata):
    err = 1
    len1 = len(Results_0to4)
    len2 = len(Results_4to8)
    len3 = len(Results_8to12)
    len4 = len(Results_12to16)
    len5 = len(Results_16to20)
    len6 = len(Results_20to24)
    len7 = len(Results_24to28)
    len8 = len(Results_28to32)
    len9 = len(Results_32to36)
    len10 = len(Results_36to40)
    len11 = len(Results_40to44)
    len12 = len(Results_44to48)
    len13 = len(Results_48to51)
    for i in range(len1):
        Z = Results_0to4[i]
        nZdiff = np.abs(Z - Zdata) **2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = i

    for i in range(len2):
        Z = Results_4to8[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + i

    for i in range(len3):
        Z = Results_8to12[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + i

    for i in range(len4):
        Z = Results_12to16[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + i

    for i in range(len5):
        Z = Results_16to20[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))
        
        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + i

    for i in range(len6):
        Z = Results_20to24[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + len5 + i

    for i in range(len7):
        Z = Results_24to28[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + len5 + len6 + i

    for i in range(len8):
        Z = Results_28to32[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + len5 + len6 + len7 + i

    for i in range(len9):
        Z = Results_32to36[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + len5 + len6 + len7 + len8 + i

    for i in range(len10):
        Z = Results_36to40[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + len5 + len6 + len7 + len8 + len9 + i

    for i in range(len11):
        Z = Results_40to44[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 + len2 + len3 + len4 + len5 + len6 + len7 + len8 + len9 + len10 + i

    for i in range(len12):
        Z = Results_44to48[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))

        if merr < err:
            err = merr
            ind = len1 +len2+len3+len4+len5+len6+len7+len8+len9+len10+len11+ i

    for i in range(len13):
        Z = Results_48to51[i]
        nZdiff = np.abs(Z - Zdata)**2 / np.abs(Zdata)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)))
    
    return ind