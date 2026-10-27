// Original KittenFish NNUE baseline. No search or move generation.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
constexpr int H=864, F=16*768, Q=64;
using Board=array<int,64>; // 0 empty; white P..K = 1..6, black = 7..12
struct Position { Board b{}; int stm=0; };
Position fen(const string& s) {
    Position p; istringstream in(s); string placement, side; in>>placement>>side;
    int rank=7,file=0; string pieces=" PNBRQKpnbrqk";
    for(char c:placement) {
        if(c=='/') { if(file!=8||rank==0) throw runtime_error("bad FEN rank"); --rank; file=0; }
        else if(c>='1'&&c<='8') {file+=c-'0'; if(file>8) throw runtime_error("bad FEN width");}
        else { auto n=pieces.find(c); if(n==string::npos||n==0||file>=8) throw runtime_error("bad FEN piece"); p.b[rank*8+file++]=int(n); }
    }
    if(rank!=0||file!=8||(side!="w"&&side!="b")) throw runtime_error("bad FEN");
    if(count(p.b.begin(),p.b.end(),6)!=1||count(p.b.begin(),p.b.end(),12)!=1) throw runtime_error("FEN requires both kings");
    if(count_if(p.b.begin(),p.b.end(),[](int x){return x!=0;})>32) throw runtime_error("too many pieces");
    p.stm=(side=="b"); return p;
}
int bucket(const Position& p,int view) {
    int k=int(find(p.b.begin(),p.b.end(),view?12:6)-p.b.begin());
    if(view) k^=56;
    return (k/16)*4+(k%8)/2;
}
int feature(int pc,int sq,int view,int bk) {
    int color=(pc-1)/6, type=(pc-1)%6;
    return bk*768+((color^view)*6+type)*64+(view?(sq^56):sq);
}
vector<int> active(const Position& p,int view) {
    vector<int> a; int bk=bucket(p,view);
    for(int s=0;s<64;s++) if(p.b[s]) a.push_back(feature(p.b[s],s,view,bk));
    return a;
}
template<class T> void writev(ostream& o,const vector<T>& v) { o.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)); }
template<class T> void readv(istream& i,vector<T>& v) { i.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T)); if(!i) throw runtime_error("truncated model"); }
struct FloatNet {
    vector<float> w=vector<float>(size_t(F)*H), b=vector<float>(H), out=vector<float>(2*H); float bias=0;
    FloatNet() { mt19937 rng(42); normal_distribution<float> d(0,.015f); for(auto& x:w)x=d(rng); for(auto& x:out)x=d(rng); }
    array<array<float,H>,2> accum(const Position& p) const {
        array<array<float,H>,2> a{};
        for(int v=0;v<2;v++) {copy(b.begin(),b.end(),a[v].begin()); for(int f:active(p,v)) for(int h=0;h<H;h++)a[v][h]+=w[size_t(f)*H+h];}
        return a;
    }
    float predict(const Position& p) const { auto a=accum(p); float y=bias; for(int v=0;v<2;v++) for(int h=0;h<H;h++) y+=out[v*H+h]*clamp(a[p.stm^v][h],0.f,1.f); return y; }
    void step(const Position& p,float target,float lr) {
        auto a=accum(p); float y=bias; for(int v=0;v<2;v++)for(int h=0;h<H;h++)y+=out[v*H+h]*clamp(a[p.stm^v][h],0.f,1.f);
        float g=clamp(2*(y-target),-10.f,10.f); array<array<float,H>,2> da{};
        for(int v=0;v<2;v++)for(int h=0;h<H;h++) {float z=a[p.stm^v][h]; da[p.stm^v][h]=(z>0&&z<1)?g*out[v*H+h]:0; out[v*H+h]=clamp(out[v*H+h]-lr*g*clamp(z,0.f,1.f),-8.f,8.f);}
        bias=clamp(bias-lr*g,-8.f,8.f);
        for(int h=0;h<H;h++) b[h]=clamp(b[h]-lr*(da[0][h]+da[1][h]),-8.f,8.f);
        // A feature shared by both perspectives correctly receives both gradients.
        for(int v=0;v<2;v++)for(int f:active(p,v))for(int h=0;h<H;h++) {auto& x=w[size_t(f)*H+h]; x=clamp(x-lr*da[v][h],-8.f,8.f);}
    }
    void save(const string& path) const {ofstream o(path,ios::binary); o.write("KFF1",4);writev(o,w);writev(o,b);writev(o,out);o.write(reinterpret_cast<const char*>(&bias),4);if(!o)throw runtime_error("save failed");}
    void load(const string& path) {ifstream i(path,ios::binary);char magic[4];i.read(magic,4);if(!i||string(magic,4)!="KFF1")throw runtime_error("wrong float model");readv(i,w);readv(i,b);readv(i,out);i.read(reinterpret_cast<char*>(&bias),4);if(!i)throw runtime_error("truncated bias");}
};
struct QuantNet {
    vector<int16_t>w=vector<int16_t>(size_t(F)*H),b=vector<int16_t>(H),out=vector<int16_t>(2*H); int32_t bias=0;
    struct State { Position p; alignas(64) array<array<int32_t,H>,2>a{}; };
    void refresh(State& s,int v) const {copy(b.begin(),b.end(),s.a[v].begin());for(int f:active(s.p,v))for(int h=0;h<H;h++)s.a[v][h]+=w[size_t(f)*H+h];}
    State start(const Position& p) const {State s;s.p=p;refresh(s,0);refresh(s,1);return s;}
    // The caller supplies a final board; captures, promotion, castling and EP
    // are all handled as square changes. This is NOT a legality checker.
    void update(State& s,const Position& next) const {
        Position old=s.p;s.p=next;
        for(int v=0;v<2;v++) {
            int bk=bucket(old,v); if(bk!=bucket(next,v)) {refresh(s,v);continue;}
            for(int sq=0;sq<64;sq++)if(old.b[sq]!=next.b[sq]) {
                if(old.b[sq]) {int f=feature(old.b[sq],sq,v,bk);for(int h=0;h<H;h++)s.a[v][h]-=w[size_t(f)*H+h];}
                if(next.b[sq]) {int f=feature(next.b[sq],sq,v,bk);for(int h=0;h<H;h++)s.a[v][h]+=w[size_t(f)*H+h];}
            }
        }
    }
    double eval(const State& s) const {int64_t y=bias;for(int v=0;v<2;v++)for(int h=0;h<H;h++)y+=int64_t(out[v*H+h])*clamp(s.a[s.p.stm^v][h],0,Q);return double(y)*400/(Q*Q);}
    void convert(const FloatNet& n) {for(size_t i=0;i<w.size();i++)w[i]=int16_t(lround(clamp(n.w[i],-8.f,8.f)*Q));for(int i=0;i<H;i++)b[i]=int16_t(lround(n.b[i]*Q));for(int i=0;i<2*H;i++)out[i]=int16_t(lround(n.out[i]*Q));bias=int32_t(lround(n.bias*Q*Q));}
    void save(const string& path)const {ofstream o(path,ios::binary);o.write("KFQ1",4);writev(o,w);writev(o,b);writev(o,out);o.write(reinterpret_cast<const char*>(&bias),4);if(!o)throw runtime_error("save failed");}
    void load(const string& path) {ifstream i(path,ios::binary);char m[4];i.read(m,4);if(!i||string(m,4)!="KFQ1")throw runtime_error("wrong quantized model");readv(i,w);readv(i,b);readv(i,out);i.read(reinterpret_cast<char*>(&bias),4);if(!i)throw runtime_error("truncated bias");}
};
struct Sample {Position p;float target;};
vector<Sample> dataset(const string& path) {
    ifstream in(path);if(!in)throw runtime_error("cannot open dataset"); vector<Sample>d;string line;size_t row=0;
    while(getline(in,line)) {++row;if(line.empty()||line[0]=='#')continue;auto t=line.find('\t');if(t==string::npos)throw runtime_error("expected FEN<TAB>white_cp at row "+to_string(row));try {Position p=fen(line.substr(0,t));size_t used;float cp=stof(line.substr(t+1),&used);if(!isfinite(cp)||line.substr(t+1).find_first_not_of(" \r",used)!=string::npos)throw runtime_error("invalid score");d.push_back({p,clamp(cp,-2000.f,2000.f)*(p.stm?-1.f:1.f)/400.f});}catch(const exception&e){throw runtime_error("row "+to_string(row)+": "+e.what());}}
    if(d.empty())throw runtime_error("empty dataset");
    return d;
}

float material(const Position& p) {
    constexpr int value[6]={100,320,330,500,900,0};
    int cp=0;
    for(int piece:p.b) if(piece) cp+=value[(piece-1)%6]*(piece<=6?1:-1);
    return float(cp)*(p.stm?-1:1);
}
struct Metrics {
    double mse=0,mae=0,pred_mean=0,pred_std=0,target_std=0,corr=0;
};
template<class Predict> Metrics metrics(const vector<Sample>& data,Predict predict,size_t limit=0) {
    size_t count=limit?min(limit,data.size()):data.size();
    double sp=0,st=0,spp=0,stt=0,spt=0; Metrics m;
    for(size_t i=0;i<count;i++) {
        const auto& sample=data[i]; double p=predict(sample.p),t=sample.target*400;
        if(!isfinite(p)) throw runtime_error("non-finite prediction");
        m.mse+=(p-t)*(p-t);m.mae+=abs(p-t);
        sp+=p;st+=t;spp+=p*p;stt+=t*t;spt+=p*t;
    }
    double n=double(count),vp=max(0.,spp/n-(sp/n)*(sp/n)),vt=max(0.,stt/n-(st/n)*(st/n));
    m.mse/=n;m.mae/=n;m.pred_mean=sp/n;m.pred_std=sqrt(vp);m.target_std=sqrt(vt);
    m.corr=vp>0&&vt>0?(spt/n-(sp/n)*(st/n))/sqrt(vp*vt):0;
    return m;
}
void print_metrics(const string& label,const Metrics& m) {
    cout<<label<<" MSE="<<m.mse<<" cp^2 MAE="<<m.mae
        <<" cp pred_mean="<<m.pred_mean<<" pred_std="<<m.pred_std
        <<" target_std="<<m.target_std<<" cp corr="<<m.corr<<'\n';
}
void diagnose(const FloatNet& n,const vector<Sample>& data) {
    print_metrics("zero baseline",metrics(data,[](const Position&){return 0.;}));
    print_metrics("material baseline",metrics(data,[](const Position& p){return material(p);}));
    print_metrics("float model",metrics(data,[&](const Position& p){return n.predict(p)*400;}));
    QuantNet q;q.convert(n);
    print_metrics("quantized model",metrics(data,[&](const Position& p){return q.eval(q.start(p));}));
    size_t low=0,high=0,total=0;
    for(size_t i=0;i<min(size_t(256),data.size());i++) {
        auto a=n.accum(data[i].p);
        for(const auto& side:a)for(float x:side){low+=x<=0;high+=x>=1;++total;}
    }
    cout<<"activation sample (up to 256 positions): clipped_low="<<100.*low/total
        <<"% clipped_high="<<100.*high/total<<"%\n";
}
const string START="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
int main(int argc,char**argv)try {
    if(argc<2)throw runtime_error("commands: train TRAIN.tsv VAL.tsv PREFIX EPOCHS LR [RESUME.kff] | export MODEL.kff MODEL.nnue | eval MODEL.nnue FEN | bench MODEL.nnue [ITERATIONS] | diagnose MODEL.kff DATA.tsv | selftest");
    string cmd=argv[1];
    if(cmd=="train"&&argc>=7) {
        auto tr=dataset(argv[2]),va=dataset(argv[3]);FloatNet n;if(argc>=8)n.load(argv[7]);int epochs=stoi(argv[5]);float lr=stof(argv[6]);if(epochs<1||!isfinite(lr)||lr<=0||lr>1)throw runtime_error("invalid epochs/lr");mt19937 rng(42);double best=1e100;
        print_metrics("zero baseline",metrics(va,[](const Position&){return 0.;}));
        print_metrics("material baseline",metrics(va,[](const Position& p){return material(p);}));
        for(int e=1;e<=epochs;e++) {
            auto started=chrono::steady_clock::now();
            shuffle(tr.begin(),tr.end(),rng);
            for(const auto& sample:tr)n.step(sample.p,sample.target,lr);
            auto validation=metrics(va,[&](const Position& p){return n.predict(p)*400;});
            cout<<"epoch "<<e<<" ";print_metrics("val",validation);
            print_metrics("train sample (up to 4096)",metrics(tr,[&](const Position& p){return n.predict(p)*400;},4096));
            n.save(string(argv[4])+".last.kff");
            if(validation.mse<best) {
                best=validation.mse;n.save(string(argv[4])+".best.kff");
                QuantNet q;q.convert(n);q.save(string(argv[4])+".nnue");
                print_metrics("best quantized val",metrics(va,[&](const Position& p){return q.eval(q.start(p));}));
            }
            cout<<"epoch seconds="<<chrono::duration<double>(chrono::steady_clock::now()-started).count()<<'\n';
        }
    } else if(cmd=="export"&&argc==4){FloatNet n;n.load(argv[2]);QuantNet q;q.convert(n);q.save(argv[3]);}
    else if(cmd=="eval"&&argc==4){QuantNet n;n.load(argv[2]);cout<<n.eval(n.start(fen(argv[3])))<<" cp (side to move)\n";}
    else if(cmd=="diagnose"&&argc==4){FloatNet n;n.load(argv[2]);auto d=dataset(argv[3]);diagnose(n,d);}
    else if(cmd=="bench"&&argc>=3){QuantNet n;n.load(argv[2]);int it=argc>3?stoi(argv[3]):1000000;if(it<1)throw runtime_error("iterations must be positive");Position p=fen(START),p2=fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");auto s=n.start(p);volatile double sink=0;auto t=chrono::steady_clock::now();for(int i=0;i<it;i++){n.update(s,(i&1)?p:p2);sink=sink+n.eval(s);}double sec=chrono::duration<double>(chrono::steady_clock::now()-t).count();cout<<it/sec<<" incremental update+eval/s, "<<sec<<" s; checksum "<<sink<<"\n";t=chrono::steady_clock::now();for(int i=0;i<it;i++)sink=sink+n.eval(s);sec=chrono::duration<double>(chrono::steady_clock::now()-t).count();cout<<it/sec<<" cached-accumulator eval/s; checksum "<<sink<<"\n";}
    else if(cmd=="selftest") {
        FloatNet f;QuantNet q;q.convert(f);auto p=fen(START);auto s=q.start(p);mt19937 rng(7);
        for(int i=0;i<200;i++){Position next=s.p;int a=int(rng()%64),b=int(rng()%64);swap(next.b[a],next.b[b]);next.stm^=1;q.update(s,next);auto fresh=q.start(next);if(s.a!=fresh.a||q.eval(s)!=q.eval(fresh))throw runtime_error("incremental mismatch");}
        // Check analytic gradients against finite differences for all layer types.
        int unit=0;auto initial=f.accum(p);
        while(unit<H && !(initial[0][unit]>.02f&&initial[0][unit]<.9f))++unit;
        if(unit==H)throw runtime_error("no finite-difference test unit");
        int feat=active(p,0).front();
        array<float*,3> parameters={&f.w[size_t(feat)*H+unit],&f.b[unit],&f.out[unit]};
        array<float,3> original{},numerical{};
        constexpr float epsilon=.001f,rate=.001f,target=.3f;
        auto loss=[&](){double error=f.predict(p)-target;return error*error;};
        for(int i=0;i<3;i++) {
            float& parameter=*parameters[i];original[i]=parameter;
            parameter=original[i]+epsilon;double plus=loss();
            parameter=original[i]-epsilon;double minus=loss();
            parameter=original[i];numerical[i]=float((plus-minus)/(2*epsilon));
        }
        f.step(p,target,rate);
        for(int i=0;i<3;i++) {
            float analytic=(original[i]-*parameters[i])/rate;
            if(abs(analytic-numerical[i])>.002f)
                throw runtime_error("finite-difference gradient mismatch");
        }
        double before=f.predict(p);f.step(p,1,.0001f);if(abs(f.predict(p)-1)>=abs(before-1))throw runtime_error("gradient did not lower loss");q.save("test.nnue");QuantNet loaded;loaded.load("test.nnue");if(q.eval(q.start(p))!=loaded.eval(loaded.start(p)))throw runtime_error("roundtrip mismatch");remove("test.nnue");cout<<"PASS: incremental rebuild equivalence, finite-difference gradients, SGD loss, quantized roundtrip\n";
    } else throw runtime_error("invalid command/arguments (train requires explicit LR)");
    return 0;
}catch(const exception&e){cerr<<"error: "<<e.what()<<'\n';return 1;}
