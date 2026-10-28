// KFD1 bounded-memory reference trainer. Included after metrics definitions.
#include <filesystem>
#include <functional>
void packed_rows(const string& list, const function<void(const Sample&)>& visit) {
    ifstream files(list); if(!files) throw runtime_error("cannot open shard list");
    string path; size_t total=0;
    while(getline(files,path)) {
        if(path.empty()||path[0]=='#')continue;
        auto full=filesystem::path(list).parent_path()/path;
        ifstream in(full,ios::binary);if(!in)throw runtime_error("cannot open shard: "+full.string());
        array<unsigned char,67> bytes;
        while(in.read(reinterpret_cast<char*>(bytes.data()),bytes.size())) {
            Sample s{};int white=0,black=0,pieces=0;
            for(int i=0;i<64;i++) {int pc=bytes[i];if(pc>12)throw runtime_error("bad packed piece");s.p.b[i]=pc;white+=pc==6;black+=pc==12;pieces+=pc!=0;}
            if(white!=1||black!=1||pieces>32||bytes[64]>1)throw runtime_error("bad packed board");
            s.p.stm=bytes[64];int cp=int(bytes[65])+(int(bytes[66])<<8);if(cp>=32768)cp-=65536;
            s.target=clamp(float(cp),-2000.f,2000.f)*(s.p.stm?-1.f:1.f)/400;
            visit(s);++total;
        }
        if(in.gcount()!=0||!in.eof())throw runtime_error("truncated packed shard");
    }
    if(!total)throw runtime_error("empty shard list");
}
void stream_train(const string& train,const string& val,const string& prefix,int epochs,float lr,const string& resume) {
    if(epochs<1||!isfinite(lr)||lr<=0||lr>1)throw runtime_error("invalid epochs/lr");
    FloatNet n;if(!resume.empty())n.load(resume);
    mt19937 rng(42);double best=1e100;
    for(int e=0;e<epochs;e++) {
        vector<Sample> buffer;buffer.reserve(8192);
        auto flush=[&](){shuffle(buffer.begin(),buffer.end(),rng);for(const auto&s:buffer)n.step(s.p,s.target,lr);buffer.clear();};
        packed_rows(train,[&](const Sample&s){buffer.push_back(s);if(buffer.size()==8192)flush();});flush();
        double mse=0;size_t count=0;
        packed_rows(val,[&](const Sample&s){double d=(n.predict(s.p)-s.target)*400;mse+=d*d;++count;});mse/=count;
        string tmp=prefix+".last.tmp";n.save(tmp);filesystem::rename(tmp,prefix+".last.kff");
        if(mse<best){best=mse;n.save(prefix+".best.kff");QuantNet q;q.convert(n);q.save(prefix+".nnue");}
        cout<<"epoch "<<e+1<<" val_mse_cp2="<<mse<<'\n';
    }
}
