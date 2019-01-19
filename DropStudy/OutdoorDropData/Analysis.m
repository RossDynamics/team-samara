%Data Analysis for Samara Drops
clear all

%Import raw data workspace
load('rawdataworkspace.mat');

%Convert inch measurements to cm
DistSM25 = DistSM25*2.54;
DistSM26 = DistSM26*2.54;
DistSM27 = DistSM27*2.54;

DistNM25 = DistNM25*2.54;
DistNM26 = DistNM26*2.54;
DistNM27 = DistNM27*2.54;

DistN26 = DistN26*2.54;
DistN27 = DistN27*2.54;

%Convert degree measurements to radians
AngSM25 = AngSM25*(pi/180);
AngSM26 = AngSM26*(pi/180);
AngSM27 = AngSM27*(pi/180);

AngNM25 = AngNM25*(pi/180);
AngNM26 = AngNM26*(pi/180);
AngNM27 = AngNM27*(pi/180);

AngN26 = AngN26*(pi/180);
AngN27 = AngN27*(pi/180);

%Remove NaN values
ind = find(isnan(DistSM25));
DistSM25(ind) = [];
ind = find(isnan(AngSM25));
AngSM25(ind) = [];

ind = find(isnan(DistNM25));
DistNM25(ind) = [];
ind = find(isnan(AngNM25));
AngNM25(ind) = [];

ind = find(isnan(DistSM26));
DistSM26(ind) = [];
ind = find(isnan(AngSM26));
AngSM26(ind) = [];

ind = find(isnan(DistNM26));
DistNM26(ind) = [];
ind = find(isnan(AngNM26));
AngNM26(ind) = [];

ind = find(isnan(DistN26));
DistN26(ind) = [];
ind = find(isnan(AngN26));
AngN26(ind) = [];

ind = find(isnan(DistSM27));
DistSM27(ind) = [];
ind = find(isnan(AngSM27));
AngSM27(ind) = [];

ind = find(isnan(DistNM27));
DistNM27(ind) = [];
ind = find(isnan(AngNM27));
AngNM27(ind) = [];

ind = find(isnan(DistN27));
DistN27(ind) = [];
ind = find(isnan(AngN27));
AngN27(ind) = [];


%Marker colors
SMcol = [0.913, 0.839, 0.086];
NMcol = [0.078, 0.686, 0.050];
Ncol = [0.976, 0.552, 0.203];

%First Plot - April 25th
figure
polarplot(AngSM25,DistSM25,'o','Color',SMcol);
hold on
polarplot(AngNM25,DistNM25,'x','Color',NMcol)
hold on
polarplot([pi-15*(pi/180), 0,-15*(pi/180)],[1500,0,1500],'k')
legend('3D Printed Silver Maple','2D Printed Norway Maple','Crane Direction')
title('April 25th')
Ap25 = gca;
Ap25.ThetaDir = 'clockwise';
Ap25.ThetaZeroLocation = 'left';


%Second Plot - April 26th
figure
polarplot(AngSM26,DistSM26,'o','Color',SMcol)
hold on
polarplot(AngNM26,DistNM26,'x','Color',NMcol)
hold on
polarplot(AngN26,DistN26,'*','Color',Ncol)
hold on
polarplot([pi-15*(pi/180), 0,-15*(pi/180)],[750,0,750],'k')
legend('3D Printed Silver Maple','2D Printed Norway Maple','Natural Norway Maple','Crane Direction')
title('April 26th')
Ap26 = gca;
Ap26.ThetaDir = 'clockwise';
Ap26.ThetaZeroLocation = 'left';

%Third Plot - April 27th
figure
polarplot(AngSM27,DistSM27,'o','Color',SMcol)
hold on
polarplot(AngNM27,DistNM27,'x','Color',NMcol)
hold on
polarplot(AngN27,DistN27,'*','Color',Ncol)
hold on
polarplot([pi-15*(pi/180), 0,-15*(pi/180)],[1500,0,1500],'k')
legend('3D Printed Silver Maple','2D Printed Norway Maple','Natural Norway Maple','Crane Direction')
title('April 27th')
Ap27 = gca;
Ap27.ThetaDir = 'clockwise';
Ap27.ThetaZeroLocation = 'left';

%Fourth Plot - Combined
figure
polarplot([AngSM25;AngSM26;AngSM27],[DistSM25;DistSM26;DistSM27],'o','Color',SMcol)
hold on
polarplot([AngNM25;AngNM26;AngNM27],[DistNM25;DistNM26;DistNM27],'x','Color',NMcol)
hold on
polarplot([AngN26;AngN27],[DistN26;DistN27],'*','Color',Ncol)
hold on
polarplot([pi-15*(pi/180), 0,-15*(pi/180)],[1500,0,1500],'k')
legend('3D Printed Silver Maple','2D Printed Norway Maple','Natural Norway Maple','Crane Direction')
title('Combined Days')
Comb = gca;
Comb.ThetaDir = 'clockwise';
Comb.ThetaZeroLocation = 'left';
