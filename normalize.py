listTotal = []
for idx in tqdm.tqdm(dfTotal.idx.drop_duplicates()):
    dfTemp = dfTotal.loc[dfTotal.idx == idx, :].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all').dropna(axis=1, how='all')
    # inf 제거
    if dfTemp.shape[0] < 5: continue
    tensorX = torch.tensor(dfTemp.iloc[:-2, 1:-1].values, dtype=torch.float32)
    tensorX = torch.nan_to_num(tensorX.abs(), nan=20, posinf=20, neginf=20)
    tensorX = torch.log1p(tensorX)/torch.log1p(torch.tensor(20, dtype=torch.float32))
    tensorCoords = torch.tensor(dfTemp.iloc[-2:, 1:-1].values, dtype=torch.float32).permute(1, 0).repeat(tensorX.shape[0], 1 ,1)
    # print(tensorX.shape, tensorCoords.shape)
    tensorX = torch.cat([tensorX.unsqueeze(2), tensorCoords], dim=2)
    listTotal.append(tensorX)
    
